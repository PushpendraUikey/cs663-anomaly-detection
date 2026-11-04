import pytest
import torch

from anomaly.padim import FeatureExtractor

# weights=None: random weights, so tests are fast and need no download.
# Shapes and plumbing do not depend on the weight values.


def test_all_layers_give_448_channels_on_56_grid():
    fx = FeatureExtractor(weights=None)
    out = fx(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 448, 56, 56)
    assert fx.out_channels == 448


@pytest.mark.parametrize("layers, channels", [
    (("layer1",), 64),
    (("layer1", "layer2"), 192),
])
def test_layer_subsets(layers, channels):
    out = FeatureExtractor(layers, weights=None)(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, channels, 56, 56)


def test_grid_is_quarter_of_input():
    out = FeatureExtractor(weights=None)(torch.randn(1, 3, 256, 256))
    assert out.shape[-2:] == (64, 64)


def test_frozen_and_stays_in_eval_mode():
    fx = FeatureExtractor(weights=None).train()
    assert not fx.net.training
    assert not any(p.requires_grad for p in fx.parameters())
    assert not fx(torch.randn(1, 3, 64, 64)).requires_grad


def test_unknown_layer_raises():
    with pytest.raises(ValueError):
        FeatureExtractor(("layer4",), weights=None)
