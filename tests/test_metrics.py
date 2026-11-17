import numpy as np
import pytest

from anomaly.metrics import best_f1, f1_at_threshold, image_auroc, pixel_auroc, threshold_from_good

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


# ---- threshold and F1 (step 8) ----

def test_threshold_from_good_is_the_max_of_good_maps():
    good = np.array([[[0.1, 0.3], [0.2, 0.0]], [[0.25, 0.05], [0.0, 0.1]]])
    assert threshold_from_good(good) == 0.3


def test_f1_hand_computed():
    masks = np.array([[1, 1, 0, 0]])
    maps = np.array([[0.9, 0.2, 0.8, 0.1]])           # threshold 0.5: TP 1, FP 1, FN 1
    assert f1_at_threshold(masks, maps, 0.5) == pytest.approx(2 / 4)
    assert f1_at_threshold(masks, maps, 0.15) == pytest.approx(2 * 2 / (4 + 1))   # TP 2, FP 1


def test_best_f1_is_one_when_separable_and_bounds_any_threshold():
    masks = np.array([[0, 0, 1, 1]])
    assert best_f1(masks, np.array([[0.1, 0.2, 0.7, 0.9]])) == pytest.approx(1.0)
    maps = np.array([[0.1, 0.8, 0.7, 0.9]])
    assert best_f1(masks, maps) >= max(f1_at_threshold(masks, maps, t) for t in [0.1, 0.5, 0.75, 0.85])
