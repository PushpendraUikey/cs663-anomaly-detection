# Report and slide notes

Running log of decisions, results and talking points, written as each step is done.
Numbers are copied from `results/`; if a CSV is regenerated, update them here.

## Setup decisions
- **Image loading** (`src/anomaly/data.py`): grayscale categories copied to 3 channels; downsampling with
  `INTER_AREA` (box low-pass before sampling, avoids aliasing at 1024 -> 224/128); masks resized with
  nearest neighbour so they stay binary; good test images get an all-zero mask.
- **ImageNet normalisation**: mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225). Verified against
  `torchvision ResNet18_Weights.IMAGENET1K_V1.transforms()`, which also confirms resize 256 -> crop 224
  (the PaDiM paper setting).
- **Metrics**: image AUROC (one score per image) and pixel AUROC (all test pixels pooled, including
  pixels of good images). Convention: higher score = more anomalous. AUROC depends only on the ranking
  of scores, so monotone transforms (sqrt, log) do not change it; smoothing and max-pooling do.
- **Caveat for slides**: defect pixels are only ~1-6% of all pixels, so pixel AUROC can look
  optimistic (huge FPR denominator). PRO / precision-based metrics are the usual complement.

## PCA baseline (Step 3)
- Method: flatten image, subtract mean image, top-k eigen-images via thin SVD of the centred data
  matrix (rows of V^T = covariance eigenvectors, eigenvalues S^2/(N-1)); never builds the D x D
  covariance (D = 49,152 at 128 px: ~10 GB). Reconstruction x_hat = mu + C^T C (x - mu).
  Map = |x - x_hat| averaged over RGB, Gaussian sigma = 2 px (about sigma = 4 at 224); score = max of map.
- Settings: 128 x 128, sigma = 2. Script: `scripts/run_pca.py` -> `results/pca.csv`,
  `results/figures/pca_k.png`.
- **Decision: k = 50 for all categories in the method comparison.** Fixed before comparing methods and
  not tuned per category on test data (that would inflate PCA). The k sweep is reported as an ablation.

### Results at k = 50 (main comparison numbers)
| category | image AUROC | pixel AUROC | explained var |
|---|---|---|---|
| carpet | 0.632 | 0.815 | 0.553 |
| grid | 0.782 | 0.763 | 0.680 |
| leather | 0.960 | 0.990 | 0.667 |
| bottle | 0.978 | 0.949 | 0.878 |
| hazelnut | 0.923 | 0.980 | 0.832 |
| **mean** | **0.855** | **0.899** | |

Fit 3-5 s per category (CPU), inference ~1 ms per image.

### Talking points (k ablation)
- Works on aligned objects and smooth textures (bottle, leather, hazelnut). Fails on random texture
  (carpet): the first eigen-image explains only 4.6% of variance; a shift of the weave by a pixel
  decorrelates pixels, so no low-dimensional pixel subspace exists. Carpet image AUROC < 0.5 for k <= 10.
- Bias/variance trade-off in k: too few components leave normal variation in the residual (false
  alarms); too many let the subspace reconstruct the defect too. Seen in bottle pixel AUROC:
  0.949 at k = 50 -> 0.941 at k = 200, where 99.8% of training variance is explained.
- Grid still improving at k = 200: regular pattern with small shifts needs many components.
- Motivation for PaDiM: model local deep features per patch instead of whole-image pixels.

## PaDiM (Step 4)
### 4a: feature extraction (`src/anomaly/padim.py`, `FeatureExtractor`)
- Frozen torchvision ResNet-18, IMAGENET1K_V1, eval mode (BatchNorm uses stored ImageNet statistics).
  Forward pass stops after layer3, so layer4 and the classifier are never computed.
- Input 256 -> centre crop 224 (paper setting). Output grid 56 x 56 (stride 4); each grid cell is
  one 4 x 4-pixel "patch" with a 448-dim embedding: layer1 (64) + layer2 (128, upsampled x2) +
  layer3 (256, upsampled x4), bilinear.
- Theoretical receptive fields (computed by hand, see below): layer1 43 px, layer2 99 px,
  layer3 211 px. layer1 = fine local texture/edges, layer3 = context covering most of the image.
  Mixing them gives each patch both local detail and context.
  RF recursion: r_out = r_in + (kernel - 1) * (input stride). Stem: conv7 s2 -> 7, maxpool3 s2 -> 11.
  layer1 (4 conv3x3 at stride 4): 11 + 4*8 = 43. layer2: +8 (strided conv) + 3*16 = 99.
  layer3: +16 + 3*32 = 211.
- CPU: ~19 ms per image for the backbone. Full 448-channel features of 280 carpet training images
  would take 1.57 GB, which is why channels are reduced (d = 100) before storing them.
