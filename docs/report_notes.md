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

## Convolutional autoencoder (Step 5)
### 5a: architecture (`src/anomaly/ae.py`, `AutoEncoder`)
- Layout follows Bergmann et al. 2019 (Table 1): encoder of 4x4 stride-2 convs (each halves the
  side) interleaved with 3x3 stride-1 convs, 128 -> 64 -> 32 -> 16 -> 8, then an 8x8 conv with no
  padding collapses the 8x8x32 map to a 1x1x100 latent code. Decoder mirrors it with 4x4 stride-2
  transposed convs (each doubles the side). LeakyReLU(0.2) everywhere, linear latent, sigmoid output
  so the reconstruction is in [0, 1] like the input (needed for SSIM later).
- 1.05 M parameters. Compression: 49,152 input numbers -> 100 (about 490x). The bottleneck is the
  whole point: it forces the network to keep only what is common to normal images, so a defect is
  reconstructed as normal and appears in the residual.
- Link to PCA: a linear autoencoder with a k-dim code and L2 loss learns the same subspace as PCA
  with k components; the conv AE is a non-linear generalisation with shared local filters.
- Input is [0, 1] RGB at 128 x 128 (`MVTecDataset(size=128, normalize=False)`), same as PCA.
- Latent size 100 is our default (configurable); not tuned.

### 5b: SSIM loss (`ssim_map`, `ssim_loss` in `src/anomaly/ae.py`)
- Our own implementation (Wang et al. 2004 formula): local means, variances and covariance from an
  11 x 11 box filter (`avg_pool2d`, stride 1, mirror padding so the map stays H x W), constants
  c1 = 0.01^2, c2 = 0.03^2 for intensity range 1. Computed per colour channel, then averaged.
  Loss = 1 - mean SSIM. The same map gives the anomaly map later (1 - SSIM per pixel).
- Window 11 as in Bergmann et al. 2019. Difference: they train on grayscale for textures; we keep
  RGB and average the channels.
- Verified against `skimage.metrics.structural_similarity` (win 11, population covariance): max
  difference < 1e-4 away from the border (border differs only by padding convention).
- Demonstration for the slide (test): a +0.1 brightness shift and Gaussian noise of sigma 0.1 have
  the same MSE (0.010) but SSIM 0.983 vs 0.855. L2 cannot tell "same pattern, brighter" from
  "pattern destroyed"; SSIM can. This is why SSIM suits texture defects.
- Link to course: the structure term cov / (sigma_x sigma_y) is the local correlation coefficient,
  the same quantity as ZNCC used in the alignment step.

### 5c: training loop (`train_autoencoder`, `split_train_val` in `src/anomaly/ae.py`)
- Adam, lr 2e-4 (Bergmann et al. 2019), batch 32, up to 200 epochs, loss "l2" (MSE) or "ssim".
- Early stopping: 10% of the good training images are held out (seeded random split). After each
  epoch the loss on them is measured without learning; training stops when it has not improved for
  20 epochs (patience), and the weights of the best epoch are restored. No test images and no
  defective images are involved, so the stopping rule does not leak test information.
- No data augmentation (Bergmann trains on 10k random crops per texture; we use the 209-391 full
  images). Expect weaker AE results than their paper; state this on the slide.
- First real run (bottle, L2, CPU): loss 0.133 -> 0.0024 in 150 epochs, still improving slowly;
  228 s for 150 epochs (1.3 s/epoch L2, 2.1 s/epoch SSIM). Training is feasible on the laptop CPU,
  so Colab is optional.

### 5d: anomaly maps from the autoencoder (`predict_autoencoder` in `src/anomaly/ae.py`)
- Map mode "l2": (x - x_hat)^2 averaged over RGB. Map mode "ssim": 1 - local SSIM(x, x_hat), window 11.
  Then Gaussian sigma = 2 px and score = max of map: same smoothing and score rule as PCA (128 px),
  and the same (maps, scores) output as `PCAModel.predict` and `PaDiM.predict`.
- The map mode is independent of the training loss: one trained network can be scored both ways.
- Preliminary check, NOT the final numbers (bottle, L2-trained, 100 epochs, stopped by the epoch
  limit and not by early stopping, 150 s on CPU): L2 map image 0.825 / pixel 0.786; SSIM map image
  0.938 / pixel 0.855. PCA k=50 on bottle: 0.978 / 0.949; PaDiM: 0.996 / 0.981. Inference
  10-12 ms per image on CPU.
- Observation: on the same reconstruction, the SSIM map beats the squared-error map by 11 pp image
  AUROC. Final numbers come from `scripts/run_ae.py`.

### 5e: autoencoder results (`scripts/run_ae.py` -> `results/ae.csv`)
Settings: 128 px, latent 100, Adam lr 2e-4, batch 32, up to 200 epochs, patience 20, seed 0, sigma 2,
CPU. AE-L2 scored with the squared-residual map, AE-SSIM with the 1 - SSIM map.

| category | AE-L2 image | AE-SSIM image | AE-L2 pixel | AE-SSIM pixel | epochs L2 / SSIM |
|---|---|---|---|---|---|
| carpet | 0.544 | 0.571 | 0.635 | 0.785 | 200* / 101 |
| grid | 0.697 | 0.582 | 0.662 | 0.620 | 125 / 137 |
| leather | 0.888 | 0.961 | 0.975 | 0.987 | 200* / 200* |
| bottle | 0.852 | 0.968 | 0.799 | 0.946 | 165 / 200* |
| hazelnut | 0.945 | 0.880 | 0.976 | 0.977 | 200* / 174 |
| **mean** | 0.785 | 0.792 | 0.809 | 0.863 | |

\* reached the 200-epoch limit; early stopping had not triggered, so these models were still
improving and their numbers are a lower bound for this architecture.

### Method comparison (main table for the slides; means over the five categories)
| method | image AUROC | pixel AUROC | fit time per category | per test image |
|---|---|---|---|---|
| PCA, k = 50 (128 px) | 0.855 | 0.899 | 3-5 s | ~1 ms |
| AE-L2 (128 px) | 0.785 | 0.809 | 220-490 s | ~3 ms |
| AE-SSIM (128 px) | 0.792 | 0.863 | 390-930 s | ~7 ms |
| PaDiM R18 (224 px) | 0.940 | 0.971 | 13-28 s | 50-65 ms |

- SSIM helps pixel AUROC (+5.4 pp mean), most on carpet (+15 pp) and bottle (+15 pp); image AUROC
  is about equal on average (SSIM better on leather and bottle, worse on grid and hazelnut).
- Both autoencoders are below the PCA baseline on average, and fail on carpet and grid (image
  AUROC 0.54-0.70), the same textures where PCA fails. Likely reasons, not tested: only 209-391
  training images with no augmentation (Bergmann uses 10k crops per texture), a 100-number code for
  a whole 128 px image, and five of ten runs cut off by the 200-epoch limit.
- Cost: the autoencoder is the slowest to fit by far (minutes vs seconds) for the weakest results;
  PaDiM needs no training at all, only one pass through a frozen network.

## Ablations (Step 6)
### 6a: Euclidean vs Mahalanobis (`--distance euclidean` -> `results/padim_euclidean.csv`)
Same features, same 100 channels, same per-position mean; only the distance changes
(Euclidean = Mahalanobis with Sigma replaced by I).

| category | Euclid image | Mahal image | Euclid pixel | Mahal pixel |
|---|---|---|---|---|
| carpet | 0.852 | **0.973** | 0.956 | **0.984** |
| grid | 0.583 | **0.924** | 0.607 | **0.924** |
| leather | 0.928 | **0.996** | 0.975 | **0.992** |
| bottle | 0.984 | **0.996** | 0.963 | **0.981** |
| hazelnut | **0.880** | 0.809 | 0.961 | **0.975** |
| **mean** | 0.845 | **0.940** | 0.893 | **0.971** |

- The covariance is worth +9.5 pp image and +7.8 pp pixel AUROC on average. Largest effect on grid
  (image 0.58 -> 0.92): Euclidean distance is dominated by the high-variance channels, so a defect
  that shows up in low-variance channels is drowned out; Mahalanobis rescales every direction by
  its normal spread (whitening).
- Exception: hazelnut image AUROC is higher with Euclidean (0.880 vs 0.809), while its pixel AUROC
  is lower. Consistent with the hazelnut analysis: the dust specks that cause false alarms are
  rare-direction deviations that Mahalanobis amplifies (not verified directly).
- Same cost: fit and inference times are unchanged.

### 6b: feature layers (`run_padim_sweep.py --study layers` -> `results/ablation_layers.csv`, figure `ablation_layers.png`)
d = 64 random channels in all three settings (layer1 has only 64 channels), so only the layers differ.
The layer1+2+3 row is therefore not the main result (which uses d = 100).

| layers | mean image AUROC | mean pixel AUROC |
|---|---|---|
| layer1 | 0.925 | 0.955 |
| layer1+2 | 0.929 | 0.969 |
| layer1+2+3 | 0.935 | 0.963 |

- Adding layer2 helps localisation most (+1.4 pp mean pixel; carpet 0.952 -> 0.977, grid 0.879 -> 0.925).
- Adding layer3 at fixed d = 64 gives +0.6 pp image but -0.6 pp pixel AUROC (grid pixel 0.925 -> 0.902):
  with a fixed budget of 64 channels, each added layer leaves fewer channels from the finer layers.
- Per category: grid image AUROC rises steadily with depth (0.876 -> 0.914 -> 0.927); carpet image
  AUROC falls (0.972 -> 0.957 -> 0.947); bottle and leather are flat (already ~0.99).
- Paper (Table I, all 15 classes, pixel): layer1 94.8, layer2 95.7, layer3 95.7, all three 97.1.
  Differences are small in both; ours are within a single random channel draw, so do not over-read.

### 6d: eps sweep (`--study eps` -> `results/ablation_eps.csv`, figure `ablation_eps.png`)
| eps | mean image AUROC | mean pixel AUROC | grid image / pixel | hazelnut image |
|---|---|---|---|---|
| 0.001 | 0.939 | 0.975 | 0.953 / 0.946 | 0.758 |
| 0.01 (default) | 0.940 | 0.971 | 0.924 / 0.924 | 0.809 |
| 0.1 | 0.918 | 0.951 | 0.831 / 0.840 | 0.844 |
| 1 | 0.869 | 0.914 | 0.650 / 0.693 | 0.875 |

- The eps = 0.01 row reproduces `results/padim.csv` exactly (consistency check of the sweep script).
- Large eps hurts: at eps = 1 the results approach the Euclidean ablation (grid image 0.650 vs 0.583
  Euclidean; mean 0.869 vs 0.845), as expected since Sigma + eps I -> eps I when eps dominates the
  eigenvalues (79 of 100 are below 0.01 on carpet).
- Grid is the sensitive category; at eps = 0.001 its pixel AUROC is 0.946, close to the paper's
  0.949. So part of our grid gap may come from effective regularisation strength, not from
  bicubic/bilinear. Not adopted: eps stays 0.01 (paper value), not tuned on test data.
- Hazelnut image AUROC moves the opposite way (0.758 -> 0.875 as eps grows): stronger
  regularisation damps the rare-direction dust-speck false alarms, same pattern as Euclidean.

### 6c: random vs PCA reduction (`--study reduction --ds 50 100` -> `results/ablation_reduction.csv`)
PCA reduction (`PaDiM.fit_pca`): top-d eigenvectors of the 448 x 448 covariance of all training
patch vectors pooled over images and positions (one extra pass, streaming sums); every patch vector
is projected onto them. Everything after the reduction is unchanged.

| reduction, d | mean image AUROC | mean pixel AUROC | mean fit time |
|---|---|---|---|
| random, 50 | 0.926 | 0.954 | 14 s |
| random, 100 (main) | 0.940 | 0.971 | 19 s |
| PCA, 50 | 0.914 | 0.971 | 33 s |
| PCA, 100 | 0.939 | **0.980** | 37 s |

Per category at d = 100 (random -> PCA): carpet image 0.973 -> 0.996, pixel 0.984 -> 0.988;
grid image 0.924 -> 0.962, pixel 0.924 -> 0.963; leather 0.996 -> 1.000; bottle unchanged;
hazelnut image 0.809 -> 0.743 (0.646 at PCA-50), pixel 0.975 -> 0.976.

- The random-100 row reproduces `results/padim.csv` exactly.
- **Our result differs from the paper.** Paper Table II (R18, all 15 classes, pixel): Rd100 96.7 vs
  PCA100 93.5, random clearly better. Ours (5 classes): PCA is better on pixel AUROC at both d
  (PCA-50 equals random-100) and on image AUROC for four of five categories. Not explained; possible
  reasons (untested): 5 vs 15 categories, a single random seed, and the paper does not specify how
  its PCA was fitted (per position vs pooled). Report as an observation, not as a contradiction.
- Hazelnut image AUROC is the exception again (PCA much worse); mean image AUROC is therefore equal.
- Cost of PCA: about twice the fit time (second pass) and more memory (peak 1.9 GB vs 1.3 GB).
- d = 200 not run: the per-position covariances scale with d^2 and the estimated peak (~3 GB)
  exceeds the laptop's free RAM. Paper reference: Rd200 97.0 vs Rd100 96.7 (+0.3 pp).

### 6e: training-set size (`--study fraction` -> `results/ablation_fraction.csv`, figure `ablation_fraction.png`)
Seeded random subsets of train/good (`data.train_subset`); test set unchanged. d = 100, eps = 0.01.

| training images | mean image AUROC | mean pixel AUROC | grid image / pixel |
|---|---|---|---|
| 10 % (21-39 images) | 0.904 | 0.945 | 0.760 / 0.811 |
| 25 % (52-98) | 0.928 | 0.964 | 0.870 / 0.895 |
| 50 % (104-196) | 0.937 | 0.969 | 0.911 / 0.913 |
| 100 % (209-391) | 0.940 | 0.971 | 0.924 / 0.924 |
| 10-shot (10 images) | 0.834 | 0.919 | 0.490 / 0.704 |

- Diminishing returns: half of the data gives 0.937 / 0.969, within 0.3 pp of the full set.
  Even 10 % keeps mean pixel AUROC at 0.945, far above PCA (0.899) and the AEs on all data.
- 10-shot: with 10 images and d = 100, each covariance has rank at most 9, so 91 of the 100
  directions are set only by eps. Bottle (0.994) and leather (0.944) still work; grid falls to
  chance (0.490 image). Grid needs the most data in every study.
- Hazelnut image AUROC falls as data grows (0.856 at 10 % -> 0.809 at 100 %): more data gives tighter
  covariances, which make the dust-speck false alarms stronger (same pattern as small eps).
- Practical message: PaDiM is data-efficient; a few dozen good images are enough for most categories.

## Misalignment extension (Step 7)
### 7a: ZNCC and template matching (`src/anomaly/align.py`: `zncc`, `match_shift`)
- ZNCC(a, b) = sum((a - mean a)(b - mean b)) / sqrt(sum (a - mean a)^2 sum (b - mean b)^2): the
  correlation coefficient of the pixel values, in [-1, 1]. Invariant to brightness offset and
  contrast gain (zncc(a, 2a + 3) = 1), so lighting changes do not disturb the alignment.
- Template matching: the centre of the reference (border = max_shift removed) is slid over the
  test image to every integer offset in [-s, s]^2 and ZNCC is computed at each; the argmax is the
  shift. Vectorised with `numpy.lib.stride_tricks.sliding_window_view` (a view, no copies).
  Cost per call: (2s+1)^2 x template size, so it is used at coarse pyramid levels (7b).
- Same quantity as the structure term of SSIM (cov / (sigma_x sigma_y)), computed over the whole
  template instead of an 11 x 11 window.
- Tests: identity / gain-offset / inverse give 1 / 1 / -1; known shifts up to (8, 8) recovered
  exactly with score 1; a darker, lower-contrast shifted copy is still matched.
