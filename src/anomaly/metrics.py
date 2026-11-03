"""Evaluation metrics. Convention: higher score = more anomalous."""
import numpy as np
from sklearn.metrics import roc_auc_score


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
