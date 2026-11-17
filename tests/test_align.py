import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from anomaly.align import (estimate_alignment, gaussian_pyramid, match_shift, random_misalignment, warp,
                           zncc)


def _smooth_image(seed=0, size=64):
    # smooth random blobs: has structure at several scales, like a real image
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.random((size, size)), 3)


def test_zncc_identity_gain_offset_and_inverse():
    a = _smooth_image()
    assert zncc(a, a) == pytest.approx(1.0)
    assert zncc(a, 2 * a + 3) == pytest.approx(1.0)       # brightness and contrast do not matter
    assert zncc(a, -a) == pytest.approx(-1.0)
    assert abs(zncc(a, _smooth_image(seed=1))) < 0.5      # unrelated images: weak correlation


@pytest.mark.parametrize("dy, dx", [(0, 0), (3, -5), (-6, 2), (8, 8)])
def test_match_shift_recovers_a_known_shift(dy, dx):
    big = _smooth_image(size=96)
    reference = big[16:80, 16:80]                          # 64 x 64
    image = big[16 - dy:80 - dy, 16 - dx:80 - dx]          # same scene moved by (dy, dx)
    found_dy, found_dx, score = match_shift(reference, image, max_shift=8)
    assert (found_dy, found_dx) == (dy, dx)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_match_shift_ignores_brightness_change():
    big = _smooth_image(size=96)
    reference = big[16:80, 16:80]
    image = 0.5 * big[13:77, 20:84] + 0.2                  # shifted by (3, -4), darker, less contrast
    assert match_shift(reference, image, max_shift=8)[:2] == (3, -4)


# ---- 7b: warp, pyramid, rotation search ----

def _object_image(size=224, seed=0):
    # dark background with a textured disc: like an object on a plain backdrop, and the texture
    # inside is not rotation-symmetric, so the rotation is identifiable
    yy, xx = np.mgrid[:size, :size]
    disc = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (0.35 * size) ** 2
    tex = _smooth_image(seed, size)
    tex = (tex - tex.min()) / (tex.max() - tex.min())   # stretch: smoothing leaves std ~ 0.03
    # without the stretch the disc edge dominates ZNCC, and a disc looks the same at every angle
    return np.where(disc, 0.2 + 0.8 * tex, 0.05)


def test_warp_identity_and_pure_shift():
    img = _object_image()
    assert np.allclose(warp(img, 0.0), img)
    moved = warp(img, 0.0, (5, -3))
    assert np.allclose(moved[20:200, 20:200], img[15:195, 23:203])   # content moved down 5, left 3


def test_warp_rotation_round_trip():
    img = _object_image()
    back = warp(warp(img, 30.0), -30.0)
    inner = (slice(60, 164), slice(60, 164))            # away from corners filled by the border
    assert zncc(back[inner], img[inner]) > 0.99


def test_pyramid_halves_each_level():
    shapes = [p.shape for p in gaussian_pyramid(np.zeros((224, 224)), 3)]
    assert shapes == [(224, 224), (112, 112), (56, 56), (28, 28)]


@pytest.mark.parametrize("angle, shift", [(37.0, (5, -8)), (-120.0, (-10, 4)), (0.0, (12, 0))])
def test_estimate_alignment_undoes_a_known_rotation_and_shift(angle, shift):
    reference = _object_image()
    image = warp(reference, angle, shift)               # the misaligned test image
    inner = (slice(50, 174), slice(50, 174))
    assert zncc(image[inner], reference[inner]) < 0.6   # really misaligned before

    found_angle, found_shift, score = estimate_alignment(reference, image)
    error = (found_angle + angle + 180) % 360 - 180     # should be the opposite rotation
    assert abs(error) < 2.0
    aligned = warp(image, found_angle, found_shift)
    assert zncc(aligned[inner], reference[inner]) > 0.95
    assert score > 0.9


def test_random_misalignment_is_seeded_and_in_range():
    a = random_misalignment(7, 30, 16, seed=0)
    assert a == random_misalignment(7, 30, 16, seed=0)
    assert a != random_misalignment(8, 30, 16, seed=0)
    for i in range(50):
        angle, (dy, dx) = random_misalignment(i, 30, 16)
        assert -30 <= angle <= 30 and -16 <= dy <= 16 and -16 <= dx <= 16
