"""PaDiM: a Gaussian per patch position over frozen ResNet-18 features.

Step 1 of the method: turn each image into a grid of patch embeddings.
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
