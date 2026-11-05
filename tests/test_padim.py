import numpy as np
import pytest
import torch

from anomaly.padim import FeatureExtractor, PaDiM, gaussian_from_sums, random_channel_index

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


def test_random_channel_index_is_seeded_and_distinct():
    a, b = random_channel_index(448, 100, seed=0), random_channel_index(448, 100, seed=0)
    assert torch.equal(a, b)
    assert len(a.unique()) == 100 and a.min() >= 0 and a.max() < 448
    assert not torch.equal(a, random_channel_index(448, 100, seed=1))


def test_gaussian_from_sums_matches_np_cov():
    rng = np.random.default_rng(0)
    X = torch.from_numpy(rng.normal(2.0, 1.5, size=(3, 40, 5)))   # P=3 positions, N=40, d=5
    s1, s2 = X.sum(dim=1), X.transpose(1, 2) @ X
    mean, cov = gaussian_from_sums(40, s1, s2, eps=0.01)
    for p in range(3):
        expected = np.cov(X[p].numpy(), rowvar=False) + 0.01 * np.eye(5)
        assert np.allclose(cov[p].numpy(), expected, atol=1e-5)
        assert np.allclose(mean[p].numpy(), X[p].numpy().mean(axis=0), atol=1e-6)


def test_fit_shapes_and_positive_definite():
    images = torch.randn(6, 3, 64, 64)
    loader = [(images[:4], None, None), (images[4:], None, None)]   # two batches
    model = PaDiM(d=10, device="cpu", weights=None).fit(loader)
    assert model.grid == (16, 16)
    assert model.mean.shape == (256, 10) and model.cov.shape == (256, 10, 10)
    assert torch.allclose(model.cov, model.cov.transpose(1, 2))
    torch.linalg.cholesky(model.cov)   # raises if any covariance is not positive definite


def test_d_out_of_range_raises():
    with pytest.raises(ValueError):
        PaDiM(d=449, device="cpu", weights=None)


def test_mahalanobis_matches_direct_formula():
    torch.manual_seed(0)
    model = PaDiM(d=8, device="cpu", weights=None).fit([(torch.randn(20, 3, 32, 32), None, None)])
    assert torch.allclose(model.precision @ model.cov, torch.eye(8).expand(64, 8, 8), atol=1e-4)

    test = torch.randn(2, 3, 32, 32)
    dist = model.mahalanobis(test)
    assert dist.shape == (2, 8, 8)

    e = model.embed(test).double()                          # (2, 8, 8, 8): B, d, H, W
    for b, (i, j) in [(0, (0, 0)), (1, (3, 5)), (1, (7, 7))]:
        p = i * 8 + j                                       # row-major position index
        diff = e[b, :, i, j] - model.mean[p].double()
        expected = torch.sqrt(diff @ torch.linalg.inv(model.cov[p].double()) @ diff)
        assert dist[b, i, j].item() == pytest.approx(expected.item(), rel=1e-4)


def test_mahalanobis_rejects_other_input_size():
    model = PaDiM(d=8, device="cpu", weights=None).fit([(torch.randn(4, 3, 32, 32), None, None)])
    with pytest.raises(ValueError):
        model.mahalanobis(torch.randn(1, 3, 64, 64))


def test_predict_shapes_and_score_is_map_max():
    model = PaDiM(d=8, device="cpu", weights=None).fit([(torch.randn(10, 3, 32, 32), None, None)])
    maps, scores = model.predict(torch.randn(3, 3, 32, 32), sigma=1.0)
    assert maps.shape == (3, 32, 32) and scores.shape == (3,)
    assert np.allclose(scores, maps.reshape(3, -1).max(axis=1))


def test_upsampling_and_blur_never_exceed_the_patch_maximum():
    # bilinear values and Gaussian-blurred values are weighted averages (weights >= 0,
    # summing to 1) of existing values, so they can never exceed the largest patch distance
    model = PaDiM(d=8, device="cpu", weights=None).fit([(torch.randn(10, 3, 32, 32), None, None)])
    test = torch.randn(2, 3, 32, 32)
    grid_max = model.mahalanobis(test).flatten(1).max(dim=1).values.numpy()
    for sigma in (0, 2.0):
        _, scores = model.predict(test, sigma=sigma)
        assert np.all(scores <= grid_max + 1e-5)
