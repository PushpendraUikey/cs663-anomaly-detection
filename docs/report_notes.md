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

### 4b: channel selection and Gaussian fit (`PaDiM.fit`)
- d = 100 of the 448 channels chosen uniformly at random (seeded, `random_channel_index`); the same
  indices are used for training and test.
- Per position p (3136 = 56 x 56 of them): mu_p = mean, Sigma_p = unbiased sample covariance + eps I,
  eps = 0.01. This is the multivariate Gaussian MLE from the course (MLE divides by N; we use N-1 like
  the paper / np.cov; the difference is a factor 280/279).
- Memory: covariances are accumulated as running sums s1 = sum x, s2 = sum x x^T (float64), using
  Sigma = (s2 - N mu mu^T)/(N-1). Peak RAM ~1.3 GB on carpet, independent of N (the laptop has 6 GB).
  Checked against the two-pass formula on real features: max difference 1.5e-11.
- Carpet: fit 17 s on CPU for 280 images (feature extraction dominates).
- **Why eps is necessary (measured, carpet):** every one of the 3136 positions has at least one
  channel (median 4, max 9) that is exactly 0 in all 280 training images. A ReLU unit that never
  fires on carpet has zero variance, so Sigma_p is singular (smallest eigenvalue ~1e-19, i.e. 0) and
  Sigma_p^{-1} does not exist, even though N = 280 > d = 100. With eps = 0.01 the smallest eigenvalue
  is 0.01 and the median condition number is 33. Dead channels by layer: layer1 1, layer2 4,
  layer3 15 (deeper ImageNet features are more object-specific and often silent on textures).
- Consequence for scoring: if a test patch activates a channel that was always 0 in training, it
  adds x^2/eps to the squared Mahalanobis distance, so eps sets how strongly "never seen before"
  is penalised. This is what the eps sweep (1e-3 to 1) measures.

### 4c-i: Mahalanobis distance (`PaDiM.mahalanobis`)
- Precision matrices Sigma_p^{-1} computed once at fit time (float64 inversion), 3136 x 100 x 100.
- M(x_p) = sqrt((x_p - mu_p)^T Sigma_p^{-1} (x_p - mu_p)): distance from the normal mean measured in
  units of the normal spread along each direction (a Euclidean distance after whitening).
- Carpet sanity check (no smoothing yet): good images max ~10.5, mean ~5.4; first "color" defect
  image max 16.0, mean inside the defect 11.6 vs 5.7 outside. Defect patches are about 2x farther
  than normal patches. Note the overlap: max of a good image (10.5) is close to max of weaker
  defects (11.5-11.8), which is where AUROC will lose points.
- Why a normal patch has M ~ 5, not sqrt(d) = 10: without eps, M^2 of a sample from the fitted Gaussian
  averages d = 100 (chi-square with d degrees of freedom). With eps, each eigen-direction contributes
  lambda_i / (lambda_i + eps) instead of 1, i.e. eps "switches off" directions with variance below eps.
  On carpet 79 of the 100 eigenvalues are below eps = 0.01; sum lambda/(lambda+eps) has median 25.5,
  sqrt = 5.05, matching the observed ~5.4 on good test images. So eps also sets the effective
  dimension of the model (link to the eps sweep).

### 4c-ii: anomaly map and image score (`PaDiM.predict`)
- 56 x 56 Mahalanobis grid -> bilinear upsample to the 224 input size -> Gaussian blur sigma = 4
  (scipy `gaussian_filter`, the same function as the PCA baseline, so smoothing is identical across
  methods) -> image score = max of the map. Same (maps, scores) output contract as `PCAModel.predict`.
- Carpet "cut" example: grid max 22.2 -> upsampled 21.2 -> blurred 17.9. Inside-defect mean barely
  changes (11.8 -> 11.7); the max drops because blur averages isolated spikes away. The 56 grid
  shows scattered single bright cells outside the defect (noise); after blurring only the coherent
  defect blob remains. That's why the image score is taken after blurring.
- Upsampling and blurring are weighted averages with non-negative weights summing to 1, so the map
  can never exceed the largest patch distance (tested).
- Good qualitative figure for slides: input | ground truth | 56 grid | final map (regenerate in the
  viz step).

### Paper reference values (Defard et al., arXiv 2011.08785, read from the PDF)
- Setup (Sec. IV): resize 256 -> centre crop 224; maps resized with **bicubic** interpolation (we use
  bilinear, per our plan); Gaussian sigma = 4; eps = 0.01; R18 uses Rd100 (100 random dims of 448).
- Table III, PaDiM-R18-Rd100 pixel AUROC (%): carpet 98.9, grid 94.9, leather 99.1, bottle 98.1,
  hazelnut 97.7 (mean of these 5: 97.7). All classes 96.7; textures 95.6; objects 97.3.
- Image-level AUROC: Table V reports only WR50-Rd550 (95.3 all classes) and EfficientNet-B5, **not R18**.
  So there is no paper image-AUROC reference for our backbone; say so on the slide.
- Useful for ablations: Table I (layer study, R18, all classes pixel AUROC): layer1 94.8, layer2 95.7,
  layer3 95.7, layer1+2+3 summed 96.0, PaDiM-R18 (concatenated, joint Gaussian) 97.1.
  Table II (R18, all classes): Rd100 96.7, PCA100 93.5, Rd200 97.0, PCA200 95.7, all 448 97.1
  (random beats PCA reduction).
- Table VII: inference 0.23 s/image for R18-Rd100 on an i7-4710HQ CPU.

### 4c-iii: first PaDiM results (`scripts/run_padim.py` -> `results/padim.csv`)
Settings: R18, layers 1+2+3, d = 100 random (seed 0), eps = 0.01, 256 -> crop 224, sigma = 4, CPU.

| category | PCA k=50 image | PaDiM image | PCA k=50 pixel | PaDiM pixel | paper R18 pixel |
|---|---|---|---|---|---|
| carpet | 0.632 | **0.973** | 0.815 | **0.984** | 0.989 |
| grid | 0.782 | **0.924** | 0.763 | **0.924** | 0.949 |
| leather | 0.960 | **0.996** | 0.990 | **0.992** | 0.991 |
| bottle | 0.978 | **0.996** | 0.949 | **0.981** | 0.981 |
| hazelnut | **0.923** | 0.809 | 0.980 | 0.975 | 0.977 |
| **mean** | 0.855 | **0.940** | 0.899 | **0.971** | 0.977 |

- Pixel AUROC is within 0.5 pp of the paper on 4/5 categories; grid is 2.5 pp below. Mean 97.1 vs 97.7.
  Possible causes (not yet tested): bilinear vs the paper's bicubic map resizing, OpenCV INTER_AREA
  vs PIL resizing, which 100 random channels were drawn (seed).
- Biggest gain over PCA is on textures (carpet +34 pp image AUROC): local deep features capture
  texture statistics that pixel-space PCA cannot.
- **Hazelnut image AUROC 0.809, below PCA (0.923)**, although its pixel AUROC is fine (0.975).
  Analysis (figure `results/figures/padim_hazelnut_false_alarms.png`):
  - Top good-image scores (26.6, 26.3, 25.3, 24.6) are above the median of every defect type
    (crack 16.9, cut 15.6, hole 20.9, print 18.8). The six worst good images cause most of the loss.
  - In 5 of the top 6 the maximum is at **row 0, on the black background**, not on the nut. Zooming
    in shows a pair of tiny bright specks (dust/lint) at the top edge, visible in several good test
    images. PaDiM correctly finds them unusual for that position; they are just not labelled defects.
    9/40 good vs 3/70 defective images have their max in the top 10 rows.
  - 4th worst (039): hot spots at the two tips of a nut lying sideways, i.e. an orientation effect
    (link to the misalignment extension).
  - Pixel AUROC is unaffected: a few hundred hot background pixels are negligible among ~5.5 M.
  - Diagnostics only (found by looking at test data, so **not** adopted as the method): ignoring a
    16-px border gives 0.943; mean of the top 1% of pixels instead of the max gives 0.923.
    The image score stays max-of-map, as in the paper.
  - Slide message: max-of-map is sensitive to a single small out-of-distribution blob anywhere,
    including acquisition artefacts; image AUROC and pixel AUROC measure different things.
- Timing: fit 13-28 s per category, ~50-65 ms per test image on CPU (paper: 0.23 s/image on an older
  CPU). Caveat for the comparison slide: PCA runs at 128 px without crop, PaDiM at 224 after a
  256 -> 224 centre crop, so pixel AUROCs are computed on slightly different regions/resolutions.
