"""Evaluation metrics. Convention: higher score = more anomalous."""
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score


def _auroc(y_true, y_score):
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: labels {y_true.shape} vs scores {y_score.shape}")
    # roc_auc_score expects 1-D arrays; ravel flattens without copying when possible
    return float(roc_auc_score(y_true.ravel(), y_score.ravel()))


def image_auroc(labels, scores):
    """Detection AUROC.

    labels: (N,) 0 = good, 1 = defective
    scores: (N,) one anomaly score per image
    """
    return _auroc(labels, scores)


def pixel_auroc(masks, maps):
    """Localization AUROC over all test pixels pooled together.

    masks: (N, H, W) ground truth {0, 1}
    maps:  (N, H, W) anomaly maps, same resolution as the masks
    """
    return _auroc(masks, maps)


def threshold_from_good(maps):
    """Threshold chosen without any defective image: the largest map value seen on held-out
    good images (= their largest image score). Any pixel above it is hotter than anything the
    model produced on normal data, so held-out good images give an empty mask."""
    return float(np.max(maps))


def f1_at_threshold(masks, maps, threshold):
    """Pixel F1 of the mask (maps >= threshold) against the ground truth, all pixels pooled.

    F1 = 2 TP / (2 TP + FP + FN): the harmonic mean of precision and recall.
    """
    pred = np.asarray(maps) >= threshold
    truth = np.asarray(masks).astype(bool)
    tp = np.sum(pred & truth)
    denom = 2 * tp + np.sum(pred & ~truth) + np.sum(~pred & truth)
    return float(2 * tp / denom) if denom > 0 else 1.0


def best_f1(masks, maps):
    """Highest pixel F1 over all thresholds. Uses the test labels, so it is an oracle upper
    bound for comparison only, never a threshold to deploy."""
    precision, recall, _ = precision_recall_curve(np.asarray(masks).ravel(), np.asarray(maps).ravel())
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(f1.max())
