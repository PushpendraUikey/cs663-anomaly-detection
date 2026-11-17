# Unsupervised Anomaly Detection and Localization on MVTec AD

CS-663 Digital Image Processing course project, IIT Bombay, Autumn 2026.
Pushpendra Uikey (23b1023), supervised by Prof. Suyash P. Awate.

Train only on defect-free images. At test time, give one anomaly score per image (detection) and
one per pixel (localization). Methods, all core code written by us:

- **PCA baseline** (`src/anomaly/pca.py`): eigen-images, map = |x − reconstruction|.
- **Convolutional autoencoder** (`src/anomaly/ae.py`): L2 or SSIM loss, early stopping.
- **PaDiM** (`src/anomaly/padim.py`): frozen ResNet-18 features, one Gaussian per patch position,
  Mahalanobis distance.
- **Extension: misalignment robustness** (`src/anomaly/align.py`): ZNCC rotation and shift search
  over a Gaussian pyramid, then bilinear warping, compared against rotation augmentation.

The slides are in `docs/slides.pdf`, and the detailed results and discussion in
`docs/report_notes.md`.

## Main results (means over carpet, grid, leather, bottle, hazelnut)

| method | image AUROC | pixel AUROC |
|---|---|---|
| PCA, k = 50 | 0.855 | 0.899 |
| AE, L2 loss | 0.785 | 0.809 |
| AE, SSIM loss | 0.792 | 0.863 |
| PaDiM, ResNet-18 | **0.940** | **0.971** |

Alignment raises PaDiM's image AUROC on the naturally misaligned screw category from 0.751 to 0.847.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build
pip install -r requirements.txt
```

Data: download MVTec AD from https://www.mvtec.com/company/research/datasets/mvtec-ad
(CC BY-NC-SA 4.0) and extract it to `data/mvtec_ad/<category>/...`. The data is not included.

Everything runs on a CPU (6 GB RAM laptop); it uses CUDA when available.

## Reproducing every result

Each script writes a CSV to `results/` and figures to `results/figures/`. Seeds are fixed (seed 0).

| command | output | runtime (CPU) |
|---|---|---|
| `python scripts/run_pca.py` | `pca.csv`, `figures/pca_k.png` | ~1 min |
| `python scripts/run_ae.py` | `ae.csv` | ~75 min |
| `python scripts/run_padim.py` | `padim.csv` | ~3 min |
| `python scripts/run_padim.py --distance euclidean --out results/padim_euclidean.csv` | Euclidean ablation | ~3 min |
| `python scripts/run_padim_sweep.py --study layers` (also `eps`, `fraction`, `reduction --ds 50 100`) | `ablation_<study>.csv` / `.png` | 5–15 min each |
| `python scripts/run_misalign.py` | `misalign.csv`, `figures/misalign.png`, `figures/misalign_examples.png` | ~10 min |
| `python scripts/run_qualitative.py` | `threshold.csv`, `figures/qualitative_<category>.png` | ~90 min (AE retraining) |

`figures/padim_hazelnut_false_alarms.png` was made with a one-off snippet during the hazelnut analysis
and has no script.

Tests: `python -m pytest` (64 tests; synthetic data only, no MVTec needed).

## Layout

```
src/anomaly/   data.py, metrics.py, pca.py, ae.py, padim.py, align.py, viz.py
scripts/       one script per experiment (argparse flags; see each docstring)
tests/         unit and sanity tests for every module
results/       CSVs and figures
docs/          proposal, report notes, slides
```

## References

1. Defard et al., PaDiM, ICPR Workshops 2021.
2. Bergmann et al., MVTec AD, CVPR 2019.
3. Bergmann et al., SSIM autoencoders, VISAPP 2019.
4. He et al., ResNet, CVPR 2016.

Public PaDiM implementations (xiahaifeng1995, openvinotoolkit/anomalib) were used only to check
numbers; no code was copied.
