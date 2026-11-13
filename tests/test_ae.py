import pytest
import torch

from anomaly.ae import AutoEncoder


def test_shapes_through_the_bottleneck():
    model = AutoEncoder(latent_dim=100)
    x = torch.rand(2, 3, 128, 128)
    z = model.encoder(x)
    assert z.shape == (2, 100, 1, 1)          # whole image squeezed into 100 numbers
    assert model(x).shape == x.shape


def test_output_is_in_unit_range():
    out = AutoEncoder()(torch.rand(2, 3, 128, 128))
    assert out.min() >= 0 and out.max() <= 1  # sigmoid


def test_wrong_size_raises():
    with pytest.raises(ValueError):
        AutoEncoder()(torch.rand(1, 3, 224, 224))


def test_gradients_reach_every_parameter():
    model = AutoEncoder()
    x = torch.rand(2, 3, 128, 128)
    ((model(x) - x) ** 2).mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
