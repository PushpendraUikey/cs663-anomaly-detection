"""PaDiM: a Gaussian per patch position over frozen ResNet-18 features.

1. FeatureExtractor turns each image into a grid of patch embeddings.
2. PaDiM.fit keeps d of the channels and fits mean + covariance per position.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

LAYERS = ("layer1", "layer2", "layer3")
LAYER_CHANNELS = {"layer1": 64, "layer2": 128, "layer3": 256}


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
                 weights=ResNet18_Weights.IMAGENET1K_V1):
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
        return feats[:, self.idx].cpu()

    def fit(self, loader):
        """loader yields (images, labels, masks) batches of defect-free images."""
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
        return self
