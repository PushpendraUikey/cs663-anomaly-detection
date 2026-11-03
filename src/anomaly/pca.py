"""PCA baseline: eigen-images of defect-free images.

Each image is flattened to a vector x in R^D (D = H*W*C). PCA finds the
k orthonormal directions (eigen-images) along which normal images vary most.
"""
import numpy as np
from scipy.ndimage import gaussian_filter


class PCAModel:
    def __init__(self, k):
        self.k = k

    def fit(self, images):
        """images: (N, ...) array of normal images, e.g. (N, H, W, 3)."""
        n = len(images)
        if not 1 <= self.k <= n - 1:
            # centring removes one degree of freedom: at most N-1 non-zero eigenvalues
            raise ValueError(f"k must be in [1, {n - 1}] for {n} images, got {self.k}")
        self.image_shape = images.shape[1:]
        X = images.reshape(n, -1).astype(np.float32)  # (N, D), one image per row

        self.mean = X.mean(axis=0)                     # mean image, (D,)
        Xc = X - self.mean

        # Thin SVD: Xc = U diag(S) Vt.  Then Xc^T Xc = Vt^T diag(S^2) Vt, so the
        # rows of Vt are the eigenvectors of the covariance C = Xc^T Xc / (N-1),
        # with eigenvalues S^2 / (N-1). We never build the D x D matrix C.
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        self.components = Vt[:self.k]                  # (k, D), orthonormal rows
        self.eigenvalues = S**2 / (n - 1)              # all N, for variance plots
        return self

    def explained_variance_ratio(self):
        """Fraction of total variance captured by the first 1, 2, ..., N components."""
        return np.cumsum(self.eigenvalues) / self.eigenvalues.sum()

    def reconstruct(self, images):
        """Project onto the k eigen-images and map back: x_hat = mu + C^T C (x - mu)."""
        if images.shape[1:] != self.image_shape:
            raise ValueError(f"expected images of shape {self.image_shape}, got {images.shape[1:]}")
        Xc = images.reshape(len(images), -1).astype(np.float32) - self.mean
        coeffs = Xc @ self.components.T             # (N, k): coordinates in the subspace
        X_hat = coeffs @ self.components + self.mean  # (N, D): back to pixel space
        return X_hat.reshape(images.shape)

    def predict(self, images, sigma=4.0):
        """Anomaly maps (N, H, W) and image scores (N,) for (N, H, W, C) images.

        map   = |x - x_hat| averaged over colour channels, Gaussian-smoothed
        score = max of the map
        """
        residual = np.abs(images - self.reconstruct(images)).mean(axis=-1)  # (N, H, W)
        if sigma > 0:
            # sigma 0 along the batch axis: smooth within each image, never across images
            residual = gaussian_filter(residual, sigma=(0, sigma, sigma))
        scores = residual.reshape(len(residual), -1).max(axis=1)
        return residual, scores
