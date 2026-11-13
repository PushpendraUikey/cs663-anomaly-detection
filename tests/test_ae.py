import numpy as np
import pytest
import torch

from skimage.metrics import structural_similarity

from anomaly.ae import AutoEncoder, ssim_loss, ssim_map


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


# ---- SSIM ----

def test_ssim_of_identical_images_is_one():
    x = torch.rand(2, 3, 64, 64)
    assert torch.allclose(ssim_map(x, x), torch.ones(2, 64, 64), atol=1e-5)
    assert ssim_loss(x, x).item() == pytest.approx(0, abs=1e-5)


def test_ssim_matches_skimage_in_the_interior():
    rng = np.random.default_rng(0)
    a = rng.random((64, 64)).astype(np.float32)
    b = np.clip(a + 0.2 * rng.standard_normal((64, 64)), 0, 1).astype(np.float32)
    _, ref = structural_similarity(a, b, win_size=11, data_range=1.0, full=True,
                                   use_sample_covariance=False)
    ours = ssim_map(torch.from_numpy(a)[None, None], torch.from_numpy(b)[None, None])[0].numpy()
    # compare away from the border: the two libraries mirror-pad slightly differently
    assert np.allclose(ours[5:-5, 5:-5], ref[5:-5, 5:-5], atol=1e-4)


def test_ssim_tolerates_brightness_shift_more_than_noise_of_equal_mse():
    torch.manual_seed(0)
    x = 0.2 + 0.6 * torch.rand(1, 1, 64, 64)
    shifted = x + 0.1                                    # same pattern, brighter
    noisy = x + 0.1 * torch.randn_like(x)                # pattern disturbed
    mse = lambda a, b: ((a - b) ** 2).mean().item()
    assert mse(x, shifted) == pytest.approx(mse(x, noisy), rel=0.1)   # L2 cannot tell them apart
    assert ssim_loss(x, shifted) < 0.5 * ssim_loss(x, noisy)          # SSIM can


def test_ssim_loss_is_differentiable():
    x = torch.rand(1, 3, 32, 32)
    y = torch.rand(1, 3, 32, 32, requires_grad=True)
    ssim_loss(x, y).backward()
    assert y.grad is not None and y.grad.abs().sum() > 0
