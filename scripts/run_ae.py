"""Autoencoder on MVTec AD: train on train/good with an L2 or SSIM loss, evaluate AUROC.

AE-L2 is scored with the squared-residual map, AE-SSIM with the 1 - SSIM map
(Bergmann et al. 2019). Writes results/ae.csv, one row per (category, loss);
the file is rewritten after every row, so a stopped run keeps what it finished.

    python scripts/run_ae.py
    python scripts/run_ae.py --categories bottle --losses ssim --epochs 50
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # make `import anomaly` work without installing

from anomaly.ae import SIZE, AutoEncoder, predict_autoencoder, train_autoencoder  # noqa: E402
from anomaly.data import MVTecDataset  # noqa: E402
from anomaly.metrics import image_auroc, pixel_auroc  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--categories", nargs="+", default=["carpet", "grid", "leather", "bottle", "hazelnut"])
    p.add_argument("--losses", nargs="+", default=["l2", "ssim"], choices=["l2", "ssim"])
    p.add_argument("--latent", type=int, default=100, help="size of the latent code")
    p.add_argument("--epochs", type=int, default=200, help="upper limit; early stopping usually ends sooner")
    p.add_argument("--patience", type=int, default=20, help="epochs without validation improvement before stopping")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--sigma", type=float, default=2.0, help="Gaussian blur of the map, in pixels (same as PCA)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "ae.csv")
    return p.parse_args()


def load_split(cat, split):
    """Whole split in memory: images (N, 3, 128, 128) in [0, 1], labels (N,), masks (N, 128, 128)."""
    ds = MVTecDataset(cat, split, size=SIZE)          # not ImageNet-normalised: nothing pretrained here
    items = [ds[i] for i in range(len(ds))]
    images = torch.stack([img for img, _, _ in items])
    labels = np.array([y for _, y, _ in items])
    masks = torch.stack([m for _, _, m in items]).numpy()
    return images, labels, masks


def run_one(cat, loss, train, test, labels, masks, args):
    torch.manual_seed(args.seed)                      # same initial weights for every run
    model = AutoEncoder(latent_dim=args.latent)
    t0 = time.perf_counter()
    history = train_autoencoder(model, train, loss=loss, epochs=args.epochs, batch_size=args.batch_size,
                                lr=args.lr, patience=args.patience, seed=args.seed)
    t1 = time.perf_counter()
    maps, scores = predict_autoencoder(model, test, mode=loss, sigma=args.sigma)
    t2 = time.perf_counter()
    best = min(history, key=lambda r: r["val_loss"])
    return {
        "category": cat, "loss": loss,
        "image_auroc": round(image_auroc(labels, scores), 4),
        "pixel_auroc": round(pixel_auroc(masks, maps), 4),
        "fit_s": round(t1 - t0, 1),
        "ms_per_image": round(1000 * (t2 - t1) / len(labels), 1),
        "epochs_run": len(history), "best_epoch": best["epoch"], "best_val_loss": round(best["val_loss"], 5),
        "stopped_early": len(history) < args.epochs,
        "n_train": len(train), "n_test": len(labels),
        "latent": args.latent, "max_epochs": args.epochs, "patience": args.patience, "lr": args.lr,
        "sigma": args.sigma, "size": SIZE, "seed": args.seed,
        "device": str(next(model.parameters()).device),
    }


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for cat in args.categories:
        train, _, _ = load_split(cat, "train")
        test, labels, masks = load_split(cat, "test")
        for loss in args.losses:
            rows.append(run_one(cat, loss, train, test, labels, masks, args))
            print(rows[-1], flush=True)
            with open(args.out, "w", newline="") as f:      # rewrite after every row
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    for loss in args.losses:
        sel = [r for r in rows if r["loss"] == loss]
        print(f"AE-{loss}: mean image AUROC {np.mean([r['image_auroc'] for r in sel]):.3f}, "
              f"mean pixel AUROC {np.mean([r['pixel_auroc'] for r in sel]):.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
