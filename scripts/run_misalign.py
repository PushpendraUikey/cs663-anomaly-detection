"""Misalignment study (our extension): PaDiM under rotated/shifted test images, with and without
classical ZNCC pre-alignment, and against training-time augmentation.

Conditions per category (3 PaDiM fits, 5 evaluations):
    original            train as is,               test as is
    misaligned          train as is,               test randomly rotated + shifted
    misaligned+align    train aligned to the ref,  test misaligned, then aligned to the ref
    misaligned+augment  train + one randomly rotated/shifted copy of each image, test misaligned
    original+align      train aligned,             test aligned (natural misalignment: screw, metal_nut)

Writes results/misalign.csv, results/figures/misalign.png and results/figures/misalign_examples.png.
    python scripts/run_misalign.py
    python scripts/run_misalign.py --categories bottle --max-angle 15
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from anomaly.align import estimate_alignment, random_misalignment, warp  # noqa: E402
from anomaly.data import IMAGENET_MEAN, IMAGENET_STD, MVTecDataset  # noqa: E402
from anomaly.metrics import image_auroc, pixel_auroc  # noqa: E402
from anomaly.padim import PaDiM  # noqa: E402
from anomaly.viz import plot_grouped_bars  # noqa: E402
from run_padim import evaluate  # noqa: E402

# bottle is photographed in a fixed pose, so the mean image is a sharp template; the other
# objects lie at random angles, where a mean image would be a blur, so one training image is used
REFERENCE = {"bottle": "mean"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--categories", nargs="+", default=["bottle", "hazelnut", "screw", "metal_nut"])
    p.add_argument("--max-angle", type=float, default=30.0, help="test rotation uniform in +-this (degrees)")
    p.add_argument("--max-shift", type=float, default=16.0, help="test shift uniform in +-this (pixels at 224)")
    p.add_argument("--search-shift", type=int, default=48, help="alignment shift search range (pixels)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "misalign.csv")
    p.add_argument("--fig", type=Path, default=ROOT / "results" / "figures" / "misalign.png")
    p.add_argument("--examples-fig", type=Path, default=ROOT / "results" / "figures" / "misalign_examples.png")
    p.add_argument("--plot-only", action="store_true", help="redraw the figure from the existing CSV")
    return p.parse_args()


class Transformed(Dataset):
    """MVTec images (256 -> crop 224, [0, 1]) with optional misalignment and/or alignment,
    then ImageNet normalisation. Image and mask always get the same warp."""

    def __init__(self, base, misalign=None, reference=None, search_shift=48, seed=0):
        self.base, self.misalign, self.reference = base, misalign, reference
        self.search_shift, self.seed = search_shift, seed
        self.align_scores = {}                                   # index -> ZNCC after alignment

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, label, mask = self.base[i]
        img = img.permute(1, 2, 0).numpy()                       # (224, 224, 3) in [0, 1]
        mask = mask.numpy().astype(np.float32)
        if self.misalign is not None:
            angle, shift = random_misalignment(i, *self.misalign, seed=self.seed)
            img, mask = warp(img, angle, shift), warp(mask, angle, shift)
        if self.reference is not None:
            angle, shift, score = estimate_alignment(self.reference, img.mean(axis=2),
                                                     max_shift=self.search_shift)
            img, mask = warp(img, angle, shift), warp(mask, angle, shift)
            self.align_scores[i] = score
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        return img, label, torch.from_numpy((mask > 0.5).astype(np.uint8))   # re-binarise the mask


def make_reference(train_base, kind):
    """Grayscale (224, 224) template that every image is aligned to."""
    if kind == "mean":
        return np.mean([train_base[i][0].numpy().mean(axis=0) for i in range(len(train_base))], axis=0)
    return train_base[0][0].numpy().mean(axis=0)                  # first training image


def plot_examples(categories, args):
    """One row per category: reference | test image | misaligned | misaligned + aligned.
    The test image is the first defective one, fixed before looking at any result."""
    from anomaly.viz import plt
    fig, axes = plt.subplots(len(categories), 4, figsize=(8, 2 * len(categories)), squeeze=False)
    for r, cat in enumerate(categories):
        kw = dict(size=256, crop=224)
        train_base, test_base = MVTecDataset(cat, "train", **kw), MVTecDataset(cat, "test", **kw)
        ref = make_reference(train_base, REFERENCE.get(cat, "first"))
        i = next(j for j, (_, label, _) in enumerate(test_base.samples) if label == 1)
        img = test_base[i][0].permute(1, 2, 0).numpy()
        angle, shift = random_misalignment(i, args.max_angle, args.max_shift, seed=args.seed)
        moved = warp(img, angle, shift)
        a, s, score = estimate_alignment(ref, moved.mean(axis=2), max_shift=args.search_shift)
        panels = [(ref, "reference"), (img, "test"), (moved, f"misaligned ({angle:+.0f} deg)"),
                  (warp(moved, a, s), f"aligned (ZNCC {score:.2f})")]
        for ax, (im, title) in zip(axes[r], panels):
            ax.imshow(np.clip(im, 0, 1), cmap="gray" if im.ndim == 2 else None)
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(cat, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.examples_fig, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mis = (args.max_angle, args.max_shift)
    rows = []
    if args.plot_only:
        with open(args.out, newline="") as f:
            rows = [{**r, "image_auroc": float(r["image_auroc"]), "pixel_auroc": float(r["pixel_auroc"])}
                    for r in csv.DictReader(f)]
    for cat in [] if args.plot_only else args.categories:
        kw = dict(size=256, crop=224)                            # not normalised: warps work in [0, 1]
        train_base, test_base = MVTecDataset(cat, "train", **kw), MVTecDataset(cat, "test", **kw)
        ref_kind = REFERENCE.get(cat, "first")
        ref = make_reference(train_base, ref_kind)

        def T(base, **k):
            return Transformed(base, search_shift=args.search_shift, seed=args.seed, **k)

        train_sets = {
            "plain": T(train_base),
            "aligned": T(train_base, reference=ref),
            # augmentation: originals plus one random rotated/shifted copy each (a different seed
            # from the test perturbation, same range)
            "augmented": ConcatDataset([T(train_base), Transformed(train_base, misalign=mis, seed=args.seed + 1)]),
        }
        conditions = [  # (name, training set, test set)
            ("original", "plain", T(test_base)),
            ("misaligned", "plain", T(test_base, misalign=mis)),
            ("misaligned+align", "aligned", T(test_base, misalign=mis, reference=ref)),
            ("misaligned+augment", "augmented", T(test_base, misalign=mis)),
            ("original+align", "aligned", T(test_base, reference=ref)),
        ]
        models = {}
        for name, train_key, test_set in conditions:
            if train_key not in models:                          # one fit per training set
                t0 = time.perf_counter()
                model = PaDiM(seed=args.seed)
                model.fit(DataLoader(train_sets[train_key], batch_size=32))
                models[train_key] = (model, time.perf_counter() - t0)
            model, fit_s = models[train_key]
            t1 = time.perf_counter()
            labels, scores, maps, masks = evaluate(model, DataLoader(test_set, batch_size=32), sigma=4.0)
            t2 = time.perf_counter()
            rows.append({
                "category": cat, "condition": name,
                "image_auroc": round(image_auroc(labels, scores), 4),
                "pixel_auroc": round(pixel_auroc(masks, maps), 4),
                "fit_s": round(fit_s, 1), "ms_per_image": round(1000 * (t2 - t1) / len(labels), 1),
                "n_train": len(train_sets[train_key]), "n_test": len(labels),
                "mean_align_zncc": round(float(np.mean(list(test_set.align_scores.values()))), 3)
                                   if test_set.align_scores else "",
                "reference": ref_kind, "max_angle": args.max_angle, "max_shift": args.max_shift,
                "seed": args.seed,
            })
            print(rows[-1], flush=True)
            with open(args.out, "w", newline="") as f:           # rewrite after every row
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        del models
    colors = {"original": "#2a78d6", "misaligned": "#e34948", "misaligned+align": "#1baf7a",
              "misaligned+augment": "#eda100", "original+align": "#6250d6"}
    plot_grouped_bars(rows, "category", "condition", ["image_auroc", "pixel_auroc"], args.fig,
                      bar_order=list(colors), colors=colors)
    plot_examples(list(dict.fromkeys(r["category"] for r in rows)), args)
    print(f"wrote {args.out}, {args.fig} and {args.examples_fig}")


if __name__ == "__main__":
    main()
