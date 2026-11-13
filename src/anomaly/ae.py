"""Convolutional autoencoder for anomaly detection (Bergmann et al., VISAPP 2019).

Trained on defect-free images only; a defect cannot pass through the small
latent code, so it is reconstructed as "normal" and shows up in |x - x_hat|.
"""
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
