"""PaDiM on MVTec AD: fit on train/good, evaluate image and pixel AUROC per category.

Writes results/padim.csv (one row per category).

    python scripts/run_padim.py
    python scripts/run_padim.py --categories bottle --d 50 --eps 0.1
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # make `import anomaly` work without installing

from anomaly.data import MVTecDataset  # noqa: E402
from anomaly.metrics import image_auroc, pixel_auroc  # noqa: E402
from anomaly.padim import LAYERS, PaDiM  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--categories", nargs="+", default=["carpet", "grid", "leather", "bottle", "hazelnut"])
    p.add_argument("--layers", nargs="+", default=list(LAYERS), choices=LAYERS)
    p.add_argument("--d", type=int, default=100, help="number of random channels kept")
    p.add_argument("--eps", type=float, default=0.01, help="covariance regulariser eps * I")
    p.add_argument("--sigma", type=float, default=4.0, help="Gaussian blur of the map, in pixels")
    p.add_argument("--size", type=int, default=256, help="resize side before cropping")
    p.add_argument("--crop", type=int, default=224, help="centre crop side (paper setting)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0, help="seed for the random channel choice")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "padim.csv")
    return p.parse_args()


def evaluate(model, loader, sigma):
    """Run predict over a test loader; return labels, scores, maps, masks as numpy."""
    labels, scores, maps, masks = [], [], [], []
    for images, y, m in loader:
        batch_maps, batch_scores = model.predict(images, sigma=sigma)
        maps.append(batch_maps)
        scores.append(batch_scores)
        labels.append(y.numpy())
        masks.append(m.numpy())
    return (np.concatenate(labels), np.concatenate(scores),
            np.concatenate(maps), np.concatenate(masks))


def run_category(cat, args):
    data = dict(size=args.size, crop=args.crop, normalize=True)
    train = DataLoader(MVTecDataset(cat, "train", **data), batch_size=args.batch_size)
    test = DataLoader(MVTecDataset(cat, "test", **data), batch_size=args.batch_size)  # no shuffle

    model = PaDiM(layers=args.layers, d=args.d, eps=args.eps, seed=args.seed)
    t0 = time.perf_counter()
    model.fit(train)
    t1 = time.perf_counter()
    labels, scores, maps, masks = evaluate(model, test, args.sigma)
    t2 = time.perf_counter()
    return {
        "category": cat,
        "image_auroc": round(image_auroc(labels, scores), 4),
        "pixel_auroc": round(pixel_auroc(masks, maps), 4),
        "fit_s": round(t1 - t0, 2),
        "ms_per_image": round(1000 * (t2 - t1) / len(labels), 1),
        "n_train": len(train.dataset), "n_test": len(labels),
        "layers": "+".join(args.layers), "d": args.d, "eps": args.eps, "sigma": args.sigma,
        "size": args.size, "crop": args.crop, "seed": args.seed, "device": model.device,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rows = []
    for cat in args.categories:
        rows.append(run_category(cat, args))
        print(rows[-1], flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    mean_img = np.mean([r["image_auroc"] for r in rows])
    mean_pix = np.mean([r["pixel_auroc"] for r in rows])
    print(f"mean image AUROC {mean_img:.3f}, mean pixel AUROC {mean_pix:.3f}; wrote {args.out}")


if __name__ == "__main__":
    main()
