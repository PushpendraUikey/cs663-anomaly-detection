"""Step 8: threshold rule and qualitative grids, for PCA, AE-L2, AE-SSIM and PaDiM.

Threshold rule (no defective image is used): the same seeded 10 % of train/good that the
autoencoder uses for early stopping is held out from every method. Each method is fitted on the
other 90 %, and its threshold is the largest map value on the held-out good images
(metrics.threshold_from_good). The mask is map >= threshold.

Per category, writes one row per method to results/threshold.csv (pixel F1 at our threshold,
the oracle best F1 that peeks at the test masks, and how many good / defective test images are
flagged), and the figure results/figures/qualitative_<category>.png.

Examples are chosen before any result is seen: the first test image of each defect type
(alphabetical, up to --n-defect types) plus the first good test image. No cherry picking.

    python scripts/run_qualitative.py
    python scripts/run_qualitative.py --categories bottle --methods pca padim
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from anomaly.ae import AutoEncoder, predict_autoencoder, split_train_val, train_autoencoder  # noqa: E402
from anomaly.data import MVTecDataset, center_crop, list_test  # noqa: E402
from anomaly.metrics import (best_f1, f1_at_threshold, image_auroc, pixel_auroc,  # noqa: E402
                             threshold_from_good)
from anomaly.padim import PaDiM  # noqa: E402
from anomaly.pca import PCAModel  # noqa: E402
from anomaly.viz import plot_qualitative  # noqa: E402
from run_ae import load_split  # noqa: E402
from run_padim import evaluate  # noqa: E402
from run_pca import load_category  # noqa: E402

METHODS = ["pca", "ae_l2", "ae_ssim", "padim"]
NAMES = {"pca": "PCA", "ae_l2": "AE-L2", "ae_ssim": "AE-SSIM", "padim": "PaDiM"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--categories", nargs="+", default=["bottle", "hazelnut", "carpet", "grid", "leather"])
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--n-defect", type=int, default=4, help="defect types shown per category")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "threshold.csv")
    p.add_argument("--fig-dir", type=Path, default=ROOT / "results" / "figures")
    return p.parse_args()


def pick_examples(category, n_defect):
    """Indices into the test list: first image of the first n_defect defect types, then first good."""
    seen, picks, good = set(), [], None
    for i, (path, label, _) in enumerate(list_test(category)):
        kind = path.parent.name
        if label == 0:
            good = i if good is None else good
        elif kind not in seen and len(seen) < n_defect:
            seen.add(kind)
            picks.append(i)
    return picks + [good]


def run_pca(cat, fit_idx, val_idx):
    # same settings as the method comparison: 128 x 128, k = 50, sigma = 2
    train, test, labels, masks = load_category(cat, 128)
    model = PCAModel(50).fit(train[fit_idx])
    val_maps, _ = model.predict(train[val_idx], sigma=2.0)
    maps, scores = model.predict(test, sigma=2.0)
    return val_maps, maps, scores, labels, masks


def run_ae(cat, loss, seed):
    # same settings as run_ae.py; train_autoencoder holds out exactly split_train_val(n, 0.1, seed)
    train, _, _ = load_split(cat, "train")
    test, labels, masks = load_split(cat, "test")
    torch.manual_seed(seed)
    model = AutoEncoder(latent_dim=100)
    train_autoencoder(model, train, loss=loss, seed=seed)
    _, val_idx = split_train_val(len(train), 0.1, seed)
    val_maps, _ = predict_autoencoder(model, train[val_idx], mode=loss)
    maps, scores = predict_autoencoder(model, test, mode=loss)
    return val_maps, maps, scores, labels, masks


def run_padim(cat, fit_idx, val_idx, seed):
    # same settings as the method comparison: 256 -> crop 224, layers 1-3, d = 100, eps = 0.01
    data = dict(size=256, crop=224, normalize=True)
    train_set = MVTecDataset(cat, "train", **data)
    model = PaDiM(d=100, eps=0.01, seed=seed)
    model.fit(DataLoader(Subset(train_set, fit_idx), batch_size=32))
    val_maps = evaluate(model, DataLoader(Subset(train_set, val_idx), batch_size=32), 4.0)[2]
    labels, scores, maps, masks = evaluate(model, DataLoader(MVTecDataset(cat, "test", **data), batch_size=32), 4.0)
    return val_maps, maps, scores, labels, masks


def to_display(m):
    """A 128 x 128 map of the full image -> the 224 x 224 view PaDiM sees (centre 224/256 = 87.5 %)."""
    if m.shape[-1] == 224:
        return m
    return cv2.resize(center_crop(m, 112), (224, 224), interpolation=cv2.INTER_LINEAR)


def main():
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cat in args.categories:
        n_train = len(MVTecDataset(cat, "train"))
        fit_idx, val_idx = (t.sort().values.tolist() for t in split_train_val(n_train, args.val_fraction, args.seed))
        examples = pick_examples(cat, args.n_defect)
        view = MVTecDataset(cat, "test", size=256, crop=224)          # un-normalised, for display
        inputs = np.stack([view[i][0].numpy().transpose(1, 2, 0) for i in examples])
        truths = np.stack([view[i][2].numpy() for i in examples])
        row_labels = [list_test(cat)[i][0].parent.name for i in examples]

        shown = []
        for method in args.methods:
            t0 = time.perf_counter()
            if method == "pca":
                val_maps, maps, scores, labels, masks = run_pca(cat, fit_idx, val_idx)
            elif method == "padim":
                val_maps, maps, scores, labels, masks = run_padim(cat, fit_idx, val_idx, args.seed)
            else:
                val_maps, maps, scores, labels, masks = run_ae(cat, method[3:], args.seed)
            thr = threshold_from_good(val_maps)
            flagged = scores >= thr
            rows.append({
                "category": cat, "method": method, "threshold": float(f"{thr:.6g}"),
                "pixel_f1": round(f1_at_threshold(masks, maps, thr), 4),
                "oracle_f1": round(best_f1(masks, maps), 4),
                "good_flagged": round(float(flagged[labels == 0].mean()), 4),
                "defect_flagged": round(float(flagged[labels == 1].mean()), 4),
                "image_auroc": round(image_auroc(labels, scores), 4),
                "pixel_auroc": round(pixel_auroc(masks, maps), 4),
                "n_fit": len(fit_idx), "n_val": len(val_idx), "seconds": round(time.perf_counter() - t0, 1),
            })
            print(rows[-1], flush=True)
            shown.append((NAMES[method], np.stack([to_display(maps[i]) for i in examples]), thr))
            with open(args.out, "w", newline="") as f:                 # rewrite after every row
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        fig = args.fig_dir / f"qualitative_{cat}.png"
        plot_qualitative(inputs, truths, shown, fig, row_labels=row_labels)
        print(f"wrote {fig}", flush=True)


if __name__ == "__main__":
    main()
