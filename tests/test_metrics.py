import numpy as np
import pytest

from anomaly.metrics import image_auroc, pixel_auroc

LABELS = [0, 0, 1, 1]


def test_perfect_separation_is_one():
    assert image_auroc(LABELS, [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_reversed_scores_is_zero():
    assert image_auroc(LABELS, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_constant_scores_is_half():
    # every (defective, good) pair is a tie, and a tie counts 1/2
    assert image_auroc(LABELS, [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_hand_computed_value():
    # defective scores {0.35, 0.8}, good scores {0.1, 0.4}.
    # Of the 4 (defective, good) pairs, 3 rank the defective one higher:
    # 0.35 > 0.1, 0.8 > 0.1, 0.8 > 0.4 (but 0.35 < 0.4)  ->  3/4
    assert image_auroc(LABELS, [0.1, 0.4, 0.35, 0.8]) == 0.75


def test_pixel_auroc_square_defect():
    masks = np.zeros((2, 8, 8), dtype=np.uint8)
    masks[0, 2:5, 2:5] = 1
    maps = np.random.default_rng(0).uniform(0, 0.5, size=masks.shape)
    maps[masks == 1] += 1.0  # every defect pixel scores above every normal pixel
    assert pixel_auroc(masks, maps) == 1.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        pixel_auroc(np.zeros((2, 8, 8)), np.zeros((2, 4, 4)))
