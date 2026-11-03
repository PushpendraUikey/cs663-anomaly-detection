import numpy as np
import pytest

from anomaly.pca import PCAModel

rng = np.random.default_rng(0)


def test_components_are_orthonormal():
    model = PCAModel(k=5).fit(rng.normal(size=(20, 4, 4, 3)))
    C = model.components
    assert np.allclose(C @ C.T, np.eye(5), atol=1e-5)


def test_eigenvalues_match_covariance_eigendecomposition():
    # small D, so we can afford to build the covariance matrix explicitly
    X = rng.normal(size=(30, 6))
    model = PCAModel(k=3).fit(X)
    C = np.cov(X, rowvar=False)                  # (6, 6), divides by N-1
    expected = np.sort(np.linalg.eigvalsh(C))[::-1]
    assert np.allclose(model.eigenvalues[:6], expected, atol=1e-4)


def test_data_in_2d_subspace_needs_two_components():
    # every "image" is mean + a*u + b*v: all variance lies in span{u, v}
    u, v = rng.normal(size=(2, 50))
    coeffs = rng.normal(size=(40, 2))
    X = 3.0 + coeffs[:, :1] * u + coeffs[:, 1:] * v
    ratio = PCAModel(k=2).fit(X).explained_variance_ratio()
    assert ratio[1] == pytest.approx(1.0, abs=1e-5)


def test_k_too_large_raises():
    with pytest.raises(ValueError):
        PCAModel(k=10).fit(rng.normal(size=(10, 8)))


def test_k_max_reconstructs_training_set_exactly():
    # with k = N-1 the subspace contains every centred training image
    X = rng.normal(size=(12, 5, 5, 3)).astype(np.float32)
    model = PCAModel(k=11).fit(X)
    assert np.allclose(model.reconstruct(X), X, atol=1e-4)


def test_predict_localises_a_bright_square():
    # normal images: smooth gradients scaled by a random brightness (1-D subspace)
    ramp = np.linspace(0, 1, 16, dtype=np.float32)[None, :, None, None] * np.ones((1, 16, 16, 3), np.float32)
    train = rng.uniform(0.5, 1.5, size=(20, 1, 1, 1)).astype(np.float32) * ramp
    model = PCAModel(k=1).fit(train)

    test = np.concatenate([ramp, ramp.copy()])
    test[1, 4:8, 4:8] += 1.0                     # defect in the second image
    maps, scores = model.predict(test, sigma=0)

    assert maps.shape == (2, 16, 16)
    assert scores[1] > scores[0] + 0.5
    assert maps[1, 4:8, 4:8].min() > maps[1].mean()  # residual concentrated on the defect
