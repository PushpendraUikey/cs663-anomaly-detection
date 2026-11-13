"""Convolutional autoencoder for anomaly detection (Bergmann et al., VISAPP 2019).

Trained on defect-free images only; a defect cannot pass through the small
latent code, so it is reconstructed as "normal" and shows up in |x - x_hat|.
"""
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
