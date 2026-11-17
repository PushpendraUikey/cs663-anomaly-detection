"""PaDiM: a Gaussian per patch position over frozen ResNet-18 features.

1. FeatureExtractor turns each image into a grid of patch embeddings.
2. PaDiM.fit keeps d of the channels and fits mean + covariance per position.
3. PaDiM.predict scores test images: Mahalanobis grid -> upsample -> blur -> max.
"""
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

LAYERS = ("layer1", "layer2", "layer3")
LAYER_CHANNELS = {"layer1": 64, "layer2": 128, "layer3": 256}
DISTANCES = ("mahalanobis", "euclidean")
REDUCTIONS = ("random", "pca")


class FeatureExtractor(nn.Module):
    """Frozen ResNet-18 -> (B, C, H/4, W/4) patch embeddings.

    Feature maps of the chosen layers are bilinearly upsampled to the grid of
    the finest chosen layer and concatenated along channels. For a 224 x 224
    input and all three layers: (B, 64+128+256 = 448, 56, 56).
    """

    def __init__(self, layers=LAYERS, weights=ResNet18_Weights.IMAGENET1K_V1):
        super().__init__()
        unknown = set(layers) - set(LAYERS)
        if not layers or unknown:
            raise ValueError(f"layers must be a non-empty subset of {LAYERS}, got {layers}")
        # keep network order, so the first entry is always the finest grid
        self.layers = [name for name in LAYERS if name in layers]
        self.net = resnet18(weights=weights)
        self.net.eval()                  # BatchNorm uses its stored ImageNet statistics
        self.net.requires_grad_(False)   # frozen: we never train it

    @property
    def out_channels(self):
        return sum(LAYER_CHANNELS[name] for name in self.layers)

    def train(self, mode=True):
        # keep the backbone in eval mode even if someone calls .train()
        super().train(mode)
        self.net.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        """x: (B, 3, H, W), ImageNet-normalised."""
        # stem: 7x7 conv stride 2, then 3x3 max-pool stride 2 -> H/4
        h = self.net.maxpool(self.net.relu(self.net.bn1(self.net.conv1(x))))
        feats = []
        for name in LAYERS:
            h = getattr(self.net, name)(h)
            if name in self.layers:
                feats.append(h)
            if len(feats) == len(self.layers):
                break                    # layer4 and the classifier are never needed

        grid = feats[0].shape[-2:]       # e.g. 56 x 56 for layer1
        feats = [f if f.shape[-2:] == grid else
                 F.interpolate(f, size=grid, mode="bilinear", align_corners=False)
                 for f in feats]
        return torch.cat(feats, dim=1)


def random_channel_index(n_channels, d, seed=0):
    """d distinct channel indices out of n_channels, fixed by the seed, sorted."""
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_channels, generator=g)[:d].sort().values


def gaussian_from_sums(n, s1, s2, eps):
    """Mean and regularised covariance per position from running sums.

    s1 = sum_i x_i         (P, d)
    s2 = sum_i x_i x_i^T   (P, d, d)
    cov = (s2 - n mu mu^T) / (n - 1) + eps I    (unbiased, same as np.cov)
    """
    mean = s1 / n
    cov = (s2 - n * mean[:, :, None] * mean[:, None, :]) / (n - 1)
    cov = cov + eps * torch.eye(cov.shape[-1], dtype=cov.dtype)
    return mean.float(), cov.float()


class PaDiM:
    def __init__(self, layers=LAYERS, d=100, eps=0.01, seed=0, device=None,
                 weights=ResNet18_Weights.IMAGENET1K_V1, distance="mahalanobis",
                 reduction="random"):
        if reduction not in REDUCTIONS:
            raise ValueError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")
        self.reduction = reduction
        if distance not in DISTANCES:
            raise ValueError(f"distance must be one of {DISTANCES}, got {distance!r}")
        self.distance = distance
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = FeatureExtractor(layers, weights).to(self.device)
        n_channels = self.extractor.out_channels
        if not 1 <= d <= n_channels:
            raise ValueError(f"d must be in [1, {n_channels}], got {d}")
        self.idx = random_channel_index(n_channels, d, seed).to(self.device)
        self.d, self.eps = d, eps

    def embed(self, images):
        """(B, 3, H, W) normalised images -> (B, d, H/4, W/4) embeddings on CPU."""
        feats = self.extractor(images.to(self.device))
        if self.reduction == "random":
            return feats[:, self.idx].cpu()                     # keep d of the C channels
        # pca: project every patch vector onto the top-d principal directions
        centred = feats - self.pca_mean[None, :, None, None]
        return torch.einsum("dc,bchw->bdhw", self.pca_components, centred).cpu()

    def fit_pca(self, loader):
        """Principal directions of all training patch vectors (all images, all positions pooled).

        Streams s1 = sum x and s2 = sum x x^T over the C-dim patch vectors, then takes the
        top-d eigenvectors of the C x C covariance. Ablation: PCA reduction vs random channels.
        """
        n, s1, s2 = 0, 0.0, 0.0
        for images, _, _ in loader:
            f = self.extractor(images.to(self.device))          # (B, C, H, W)
            f = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])   # (B*H*W, C): one row per patch
            s1 = s1 + f.sum(dim=0).double().cpu()
            s2 = s2 + (f.T @ f).double().cpu()                  # (C, C)
            n += len(f)
        mean = s1 / n
        cov = (s2 - n * torch.outer(mean, mean)) / (n - 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)               # ascending eigenvalues, columns = vectors
        top = eigvecs[:, -self.d:].flip(dims=[1])               # top-d, largest first
        self.pca_mean = mean.float().to(self.device)            # (C,)
        self.pca_components = top.T.float().to(self.device)     # (d, C), orthonormal rows
        self.pca_explained = (eigvals[-self.d:].sum() / eigvals.sum()).item()

    def fit(self, loader):
        """loader yields (images, labels, masks) batches of defect-free images."""
        if self.reduction == "pca":
            self.fit_pca(loader)                                # first pass: learn the projection
        n, s1, s2 = 0, 0.0, 0.0
        for images, _, _ in loader:
            e = self.embed(images)                              # (B, d, H, W)
            b, d, h, w = e.shape
            # one row per position: (P, B, d) with P = H*W. float64 because
            # s2 - n mu mu^T subtracts two large, nearly equal numbers.
            e = e.reshape(b, d, h * w).permute(2, 0, 1).double()
            s1 = s1 + e.sum(dim=1)                              # (P, d)
            s2 = s2 + e.transpose(1, 2) @ e                     # (P, d, d), sum of outer products
            n += b
        if n < 2:
            raise ValueError("need at least 2 training images to estimate a covariance")
        self.grid = (h, w)
        self.mean, self.cov = gaussian_from_sums(n, s1, s2, self.eps)
        # invert once here, not for every test image; float64 for accuracy at small eps
        self.precision = torch.linalg.inv(self.cov.double()).float()  # (P, d, d)
        return self

    def _diff(self, images):
        """x_p - mu_p for every position and image: (P, B, d), plus the output shape (B, H/4, W/4)."""
        e = self.embed(images)                                  # (B, d, H, W)
        b, d, h, w = e.shape
        if (h, w) != self.grid:
            raise ValueError(f"test grid {(h, w)} differs from training grid {self.grid}")
        return e.reshape(b, d, h * w).permute(2, 0, 1) - self.mean[:, None, :], (b, h, w)

    def mahalanobis(self, images):
        """Patch-level Mahalanobis distance, (B, H/4, W/4).

        M(x_p) = sqrt( (x_p - mu_p)^T  Sigma_p^{-1}  (x_p - mu_p) ) at every position p.
        """
        diff, shape = self._diff(images)                        # (P, B, d)
        # quadratic form for all P x B vectors at once: row (diff^T Sigma^-1), times diff, summed
        m2 = ((diff @ self.precision) * diff).sum(dim=-1)       # (P, B)
        # m2 >= 0 in exact arithmetic; clamp tiny negative round-off before the square root
        return m2.clamp_min(0).sqrt().T.reshape(shape)

    def euclidean(self, images):
        """Patch-level Euclidean distance to the position mean, (B, H/4, W/4).

        E(x_p) = sqrt( (x_p - mu_p)^T (x_p - mu_p) ): Mahalanobis with Sigma_p replaced by I,
        so every channel counts equally and correlations are ignored. Ablation only.
        """
        diff, shape = self._diff(images)
        return (diff * diff).sum(dim=-1).sqrt().T.reshape(shape)

    def predict(self, images, sigma=4.0):
        """Anomaly maps (B, H, W) at input resolution and image scores (B,), as numpy.

        Same output contract as PCAModel.predict, so scripts can treat methods alike.
        """
        dist = getattr(self, self.distance)(images)              # (B, H/4, W/4)
        maps = F.interpolate(dist[:, None], size=images.shape[-2:],
                             mode="bilinear", align_corners=False)[:, 0].numpy()   # (B, H, W)
        if sigma > 0:
            # same smoothing as the PCA baseline: within each image, never across the batch
            maps = gaussian_filter(maps, sigma=(0, sigma, sigma))
        scores = maps.reshape(len(maps), -1).max(axis=1)
        return maps, scores
