"""Classical pre-alignment for misaligned test images (our extension).

Step 7a: zero-mean normalised cross-correlation (ZNCC) and template matching over shifts.
Step 7b: rotation + shift search over an image pyramid (coarse to fine), bilinear warping.
"""
import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter


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


def warp(image, angle, shift=(0.0, 0.0)):
    """Rotate `image` by `angle` degrees about its centre, then move it by shift = (dy, dx) pixels.

    Bilinear interpolation; pixels that come from outside the image copy the nearest border pixel.
    Works on (H, W) and (H, W, C) arrays. Positive angle = counter-clockwise on screen.
    """
    h, w = image.shape[:2]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    a = np.deg2rad(angle)
    cos, sin = np.cos(a), np.sin(a)
    # forward map p -> R (p - c) + c + t, written as a 2 x 3 matrix acting on (x, y, 1)
    R = np.array([[cos, sin], [-sin, cos]])
    t = np.array([cx, cy]) - R @ np.array([cx, cy]) + np.array([shift[1], shift[0]])
    M = np.hstack([R, t[:, None]]).astype(np.float64)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def gaussian_pyramid(image, levels):
    """[image, half, quarter, ...] with `levels` halvings: blur (sigma = 1) before each 2x subsampling,
    so that detail finer than the new pixel size is removed instead of aliasing."""
    pyramid = [image.astype(np.float64)]
    for _ in range(levels):
        pyramid.append(gaussian_filter(pyramid[-1], sigma=1)[::2, ::2])
    return pyramid


def estimate_alignment(reference, image, angles=np.arange(-180, 180, 10), levels=3, max_shift=32,
                       refine_shift=2):
    """Rotation and shift that make warp(image, angle, shift) look like `reference` (both (H, W) gray).

    1. Coarsest level (H / 2^levels): try every angle in `angles`; for each, find the best shift
       within +-max_shift / 2^levels with match_shift; keep the angle with the highest ZNCC.
    2. Each finer level: double the shift, try the angle and +-half the previous angle step, and
       re-search the shift within +-refine_shift pixels around the doubled estimate.
    Returns (angle in degrees, (dy, dx) in full-resolution pixels, ZNCC at full resolution).
    """
    ref_pyr = gaussian_pyramid(reference, levels)
    img_pyr = gaussian_pyramid(image, levels)

    def best_shift(level, angle, shift, s):
        rotated = warp(img_pyr[level], angle, shift)
        dy, dx, score = match_shift(ref_pyr[level], rotated, s)
        # reference content at r sits at r + dy in `rotated`, so move it back by (dy, dx)
        return (shift[0] - dy, shift[1] - dx), score

    coarse_s = max(1, round(max_shift / 2 ** levels))
    best = max(((a, *best_shift(levels, a, (0.0, 0.0), coarse_s)) for a in angles), key=lambda r: r[2])
    angle, shift, _ = best

    step = float(np.diff(angles).min()) if len(angles) > 1 else 0.0
    for level in range(levels - 1, -1, -1):
        shift = (2 * shift[0], 2 * shift[1])
        step /= 2
        candidates = [angle - step, angle, angle + step] if step > 0 else [angle]
        angle, shift, _ = max(((a, *best_shift(level, a, shift, refine_shift)) for a in candidates),
                              key=lambda r: r[2])

    score = zncc(warp(image.astype(np.float64), angle, shift), reference)
    return float(angle), (float(shift[0]), float(shift[1])), score
