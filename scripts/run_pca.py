"""PCA baseline on MVTec AD: sweep the number of components k.

Writes results/pca.csv (one row per category x k) and results/figures/pca_k.png.

    python scripts/run_pca.py
    python scripts/run_pca.py --categories bottle --ks 10 50 --size 128
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # make `import anomaly` work without installing

from anomaly.data import list_test, list_train, load_image, load_mask  # noqa: E402
from anomaly.metrics import image_auroc, pixel_auroc  # noqa: E402
from anomaly.pca import PCAModel  # noqa: E402
from anomaly.viz import plot_metric_sweep  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--categories", nargs="+", default=["carpet", "grid", "leather", "bottle", "hazelnut"])
    p.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20, 50, 100, 200])
    p.add_argument("--size", type=int, default=128, help="image side length in pixels")
    p.add_argument("--sigma", type=float, default=2.0, help="Gaussian smoothing of the map, in pixels")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "pca.csv")
    p.add_argument("--fig", type=Path, default=ROOT / "results" / "figures" / "pca_k.png")
    return p.parse_args()


def load_category(category, size):
    train = np.stack([load_image(p, size) for p in list_train(category)])
    test = list_test(category)
    images = np.stack([load_image(p, size) for p, _, _ in test])
    labels = np.array([label for _, label, _ in test])
    masks = np.stack([load_mask(m, size) for _, _, m in test])
    return train, images, labels, masks


def main():
    args = parse_args()
    np.random.seed(args.seed)  # PCA via SVD is deterministic; seeded for consistency across scripts
    rows = []
    for cat in args.categories:
        train, images, labels, masks = load_category(cat, args.size)
        for k in args.ks:
            if k > len(train) - 1:
                print(f"skip {cat} k={k}: only {len(train)} training images")
                continue
            t0 = time.perf_counter()
            model = PCAModel(k).fit(train)
            t1 = time.perf_counter()
            maps, scores = model.predict(images, sigma=args.sigma)
            t2 = time.perf_counter()
            row = {
                "category": cat, "k": k,
                "image_auroc": round(image_auroc(labels, scores), 4),
                "pixel_auroc": round(pixel_auroc(masks, maps), 4),
                "explained_var": round(float(model.explained_variance_ratio()[k - 1]), 4),
                "fit_s": round(t1 - t0, 3),
                "ms_per_image": round(1000 * (t2 - t1) / len(images), 3),
                "n_train": len(train), "n_test": len(images),
                "size": args.size, "sigma": args.sigma,
            }
            rows.append(row)
            print(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    plot_metric_sweep(rows, "k", ["image_auroc", "pixel_auroc"], args.fig,
                      log_x=True, xlabel="number of components k")
    print(f"wrote {args.out} and {args.fig}")


if __name__ == "__main__":
    main()
