"""PCA baseline: eigen-images of defect-free images.

Each image is flattened to a vector x in R^D (D = H*W*C). PCA finds the
k orthonormal directions (eigen-images) along which normal images vary most.
"""
import numpy as np


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
