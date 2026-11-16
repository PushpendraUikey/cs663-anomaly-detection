import copy

import numpy as np
import pytest
import torch

from skimage.metrics import structural_similarity

from anomaly.ae import AutoEncoder, split_train_val, ssim_loss, ssim_map, train_autoencoder


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


# ---- training ----

def _stripe_images(n=24):
    # one fixed stripe pattern with a random contrast per image: easy to learn,
    # and it has structure, so the SSIM loss has something to match
    torch.manual_seed(0)
    stripes = torch.sin(torch.arange(128) * 0.2).view(1, 1, 128, 1).expand(1, 3, 128, 128)
    return 0.5 + (0.2 + 0.2 * torch.rand(n, 1, 1, 1)) * stripes


def test_split_is_disjoint_complete_and_seeded():
    train, val = split_train_val(50, 0.1, seed=0)
    assert len(val) == 5 and len(train) == 45
    assert sorted(torch.cat([train, val]).tolist()) == list(range(50))
    assert torch.equal(val, split_train_val(50, 0.1, seed=0)[1])


@pytest.mark.parametrize("loss", ["l2", "ssim"])
def test_training_lowers_validation_loss(loss):
    torch.manual_seed(0)
    model = AutoEncoder(latent_dim=8)
    history = train_autoencoder(model, _stripe_images(), loss=loss, epochs=10, batch_size=8,
                                lr=2e-3, device="cpu")
    assert history[-1]["val_loss"] < history[0]["val_loss"]
    assert not model.training                              # returned in eval mode


def test_early_stopping_stops_and_restores_best_weights():
    torch.manual_seed(0)
    model = AutoEncoder(latent_dim=8)
    before = copy.deepcopy(model.state_dict())
    # lr = 0: weights never change, so the validation loss never improves after epoch 1
    history = train_autoencoder(model, _stripe_images(), epochs=50, batch_size=8, lr=0.0,
                                patience=3, device="cpu")
    assert len(history) == 4                               # epoch 1 is best, then 3 bad epochs
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
