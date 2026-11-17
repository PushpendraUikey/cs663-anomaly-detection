import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from anomaly.align import match_shift, zncc


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
