"""Classical pre-alignment for misaligned test images (our extension).

Step 7a: zero-mean normalised cross-correlation (ZNCC) and template matching over shifts.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def zncc(a, b):
    """ZNCC of two equal-size grayscale images: a number in [-1, 1].

    ZNCC = sum((a - mean a)(b - mean b)) / sqrt(sum (a - mean a)^2 * sum (b - mean b)^2)
    It is the correlation coefficient of the pixel values, so it ignores a change of
    brightness (offset) and contrast (gain): zncc(a, 2a + 3) = 1.
    """
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def match_shift(reference, image, max_shift):
    """Find the integer shift (dy, dx) of `image` relative to `reference` that maximises ZNCC.

    The template is the centre of `reference` with a border of max_shift removed. It is slid over
    `image` to every offset in [-max_shift, max_shift]^2, and ZNCC is computed at each offset
    (classic template matching). Returns (dy, dx, best_zncc): the content of `reference` at
    (r, c) appears in `image` at (r + dy, c + dx).
    """
    s = max_shift
    h, w = reference.shape
    template = reference[s:h - s, s:w - s].astype(np.float64)
    t = template - template.mean()
    t_norm = np.sqrt((t * t).sum())

    # every (2s+1) x (2s+1) placement of a template-sized window inside the image, without copying
    windows = sliding_window_view(image.astype(np.float64), template.shape)   # (2s+1, 2s+1, th, tw)
    win_mean = windows.mean(axis=(-2, -1), keepdims=True)
    centred = windows - win_mean
    numerator = (centred * t).sum(axis=(-2, -1))                               # (2s+1, 2s+1)
    win_norm = np.sqrt((centred * centred).sum(axis=(-2, -1)))
    scores = numerator / np.maximum(win_norm * t_norm, 1e-12)

    i, j = np.unravel_index(np.argmax(scores), scores.shape)
    return int(i) - s, int(j) - s, float(scores[i, j])
