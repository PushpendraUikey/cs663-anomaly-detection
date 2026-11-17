"""Convolutional autoencoder for anomaly detection (Bergmann et al., VISAPP 2019).

Trained on defect-free images only; a defect cannot pass through the small
latent code, so it is reconstructed as "normal" and shows up in |x - x_hat|.
"""
import copy

from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
from torch import nn

SIZE = 128   # the 8 x 8 bottleneck kernel below assumes 128 x 128 inputs


def _conv(c_in, c_out, k, s):
    # k=4, s=2, p=1 halves the side: (H + 2 - 4) / 2 + 1 = H / 2.  k=3, s=1, p=1 keeps it.
    return [nn.Conv2d(c_in, c_out, k, s, padding=1), nn.LeakyReLU(0.2)]


def _deconv(c_in, c_out):
    # transposed conv k=4, s=2, p=1 doubles the side: (H - 1) * 2 - 2 + 4 = 2H
    return [nn.ConvTranspose2d(c_in, c_out, 4, 2, padding=1), nn.LeakyReLU(0.2)]


class AutoEncoder(nn.Module):
    """(B, 3, 128, 128) in [0, 1] -> latent (B, latent_dim, 1, 1) -> (B, 3, 128, 128) in [0, 1]."""

    def __init__(self, latent_dim=100, channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            *_conv(channels, 32, 4, 2),   # 128 -> 64
            *_conv(32, 32, 4, 2),         # 64 -> 32
            *_conv(32, 32, 3, 1),         # 32
            *_conv(32, 64, 4, 2),         # 32 -> 16
            *_conv(64, 64, 3, 1),         # 16
            *_conv(64, 128, 4, 2),        # 16 -> 8
            *_conv(128, 64, 3, 1),        # 8
            *_conv(64, 32, 3, 1),         # 8
            nn.Conv2d(32, latent_dim, 8),  # 8 -> 1, no activation: a linear code
        )
        self.decoder = nn.Sequential(      # mirror image of the encoder
            nn.ConvTranspose2d(latent_dim, 32, 8), nn.LeakyReLU(0.2),   # 1 -> 8
            *_conv(32, 64, 3, 1),
            *_conv(64, 128, 3, 1),
            *_deconv(128, 64),            # 8 -> 16
            *_conv(64, 64, 3, 1),
            *_deconv(64, 32),             # 16 -> 32
            *_conv(32, 32, 3, 1),
            *_deconv(32, 32),             # 32 -> 64
            nn.ConvTranspose2d(32, channels, 4, 2, padding=1),          # 64 -> 128
            nn.Sigmoid(),                 # output in [0, 1], same range as the input
        )

    def forward(self, x):
        if x.shape[-2:] != (SIZE, SIZE):
            raise ValueError(f"expected {SIZE} x {SIZE} input, got {tuple(x.shape[-2:])}")
        return self.decoder(self.encoder(x))


def ssim_map(x, y, window=11, c1=0.01 ** 2, c2=0.03 ** 2):
    """Local structural similarity of two image batches in [0, 1].

    x, y: (B, C, H, W).  Returns (B, H, W) with values in [-1, 1]; 1 = identical window.

    Around every pixel, a window x window box gives local means, variances and the
    covariance.  SSIM = luminance term * contrast-and-structure term:

        (2 mu_x mu_y + c1) (2 cov_xy + c2)
        ------------------------------------------------
        (mu_x^2 + mu_y^2 + c1) (var_x + var_y + c2)

    c1, c2 are the standard constants (0.01 * L)^2, (0.03 * L)^2 for range L = 1;
    they keep the ratio stable where a window is dark or flat.
    """
    pad = window // 2

    def local_mean(t):
        # box filter: mirror-pad so the output keeps the H x W size, then average
        return F.avg_pool2d(F.pad(t, (pad, pad, pad, pad), mode="reflect"), window, stride=1)

    mu_x, mu_y = local_mean(x), local_mean(y)
    # var = E[x^2] - E[x]^2 and cov = E[xy] - E[x]E[y], with E = the local box mean
    var_x = local_mean(x * x) - mu_x ** 2
    var_y = local_mean(y * y) - mu_y ** 2
    cov = local_mean(x * y) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return ssim.mean(dim=1)              # average the colour channels


def ssim_loss(x, y, window=11):
    """1 - mean SSIM: 0 for identical images, larger = less similar."""
    return 1 - ssim_map(x, y, window).mean()


def split_train_val(n, val_fraction=0.1, seed=0):
    """Random, seeded split of indices 0..n-1 into (train, val); at least 1 in val."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = max(1, round(val_fraction * n))
    return perm[n_val:], perm[:n_val]


def train_autoencoder(model, images, loss="l2", epochs=200, batch_size=32, lr=2e-4,
                      val_fraction=0.1, patience=20, seed=0, device=None, log=None):
    """Train on defect-free images with early stopping; returns the per-epoch history.

    images: (N, 3, 128, 128) float tensor in [0, 1].
    loss:   "l2" (mean squared error) or "ssim" (1 - mean SSIM).
    Stops when the validation loss has not improved for `patience` epochs, and
    puts the weights of the best epoch back into the model.
    """
    loss_fn = {"l2": F.mse_loss, "ssim": lambda out, x: ssim_loss(x, out)}[loss]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_idx, val_idx = split_train_val(len(images), val_fraction, seed)
    val = images[val_idx].to(device)
    g = torch.Generator().manual_seed(seed)                 # fixes the batch order
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history, best_val, best_state, bad_epochs = [], float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=g)]   # reshuffle every epoch
        total = 0.0
        for i in range(0, len(order), batch_size):
            x = images[order[i:i + batch_size]].to(device)
            batch_loss = loss_fn(model(x), x)               # target = the input itself
            opt.zero_grad()                                 # clear gradients of the last step
            batch_loss.backward()                           # d loss / d every weight
            opt.step()                                      # move the weights downhill
            total += batch_loss.item() * len(x)
        model.eval()
        with torch.no_grad():                               # validation: measure only, never learn
            val_loss = loss_fn(model(val), val).item()
        history.append({"epoch": epoch, "train_loss": total / len(order), "val_loss": val_loss})
        if log:
            log(history[-1])

        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())  # snapshot of the best weights
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return history


@torch.no_grad()
def predict_autoencoder(model, images, mode="l2", sigma=2.0, batch_size=32, device=None):
    """Anomaly maps (N, H, W) and image scores (N,), as numpy.

    mode "l2":   map = (x - x_hat)^2 averaged over colour channels.
    mode "ssim": map = 1 - local SSIM(x, x_hat).
    Then the same Gaussian smoothing and max-of-map score as PCA and PaDiM.
    """
    device = device or next(model.parameters()).device
    model.eval()
    maps = []
    for i in range(0, len(images), batch_size):
        x = images[i:i + batch_size].to(device)
        x_hat = model(x)
        if mode == "l2":
            m = ((x - x_hat) ** 2).mean(dim=1)              # (B, H, W)
        elif mode == "ssim":
            m = 1 - ssim_map(x, x_hat)                      # (B, H, W)
        else:
            raise ValueError(f"mode must be 'l2' or 'ssim', got {mode!r}")
        maps.append(m.cpu())
    maps = torch.cat(maps).numpy()
    if sigma > 0:
        maps = gaussian_filter(maps, sigma=(0, sigma, sigma))   # within each image only
    scores = maps.reshape(len(maps), -1).max(axis=1)
    return maps, scores
