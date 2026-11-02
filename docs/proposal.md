# CS-663 Course Project Proposal: Unsupervised Anomaly Detection and Localization in Images

**Student:** Pushpendra (23b1023)
**Deadline:** 9 Nov 2026 (10-minute presentation, with code and results on Moodle)
**Topic source:** "Anomaly detection in images" on the CS-663 *Course-Project Examples* slide, which links the PaDiM paper (link.springer.com/chapter/10.1007/978-3-030-68799-1_35)

---

## 1. Problem

In industrial inspection, defect-free images are easy to collect but defective ones are rare, and defects vary widely (scratches, holes, cracks, contamination, missing parts). So we cannot train a normal supervised classifier.

The task:

- **Training:** use only normal (defect-free) images of one product or texture.
- **Testing:**
  1. **Detection:** decide whether a test image is anomalous (one score per image).
  2. **Localization:** mark where the anomaly is (one score per pixel, an "anomaly map").

## 2. Link to course material

| Course topic | Where it is used |
|---|---|
| Multivariate Gaussian, ML estimation of mean and covariance (PCA lecture) | PaDiM fits one Gaussian per patch position |
| Mahalanobis distance and the anomaly-detection slide (PCA lecture) | Anomaly score; Euclidean vs Mahalanobis comparison |
| PCA, principal subspace, PCA denoising / eigenfaces (PCA lecture) | Baseline 1: PCA reconstruction error; PCA feature reduction |
| CNNs, receptive fields, ResNet, residual connections (DNN lecture) | Pretrained ResNet-18 patch features from several layers |
| Encoder–decoder, pooling and un-pooling (DNN lecture, segmentation part) | Baseline 2: convolutional autoencoder |
| Bilinear interpolation (Basic Operations) | Upsampling feature maps and anomaly maps |
| Gaussian smoothing (Spatial Filtering) | Smoothing the anomaly map |
| Otsu thresholding (Point Operations) | Turning the anomaly map into a defect mask |
| Noise models, bilateral filter, CLAHE (Denoising, Point Operations) | Robustness study |
| SSIM | Autoencoder loss and residual map (Bergmann et al. 2019) |

**What is new beyond class:** patch-wise density modelling on deep features, one-class learning, anomaly localization, and AUROC-based evaluation of detection and segmentation.

## 3. Methods

All three methods are implemented by me. Only the pretrained ResNet-18 weights come from torchvision.

### 3.1 Baseline 1: PCA reconstruction (classical)
- Flatten the normal training images (grayscale or RGB, resized to 128×128 or 224×224) into vectors.
- Compute the mean image and the top-k eigenvectors ("eigen-images").
- For a test image x: project onto the subspace, reconstruct x̂ = μ + V Vᵀ(x − μ).
- **Anomaly map:** |x − x̂| per pixel, Gaussian-smoothed.
- **Image score:** maximum (or mean) of the map.
- **Ablation:** number of components k.

### 3.2 Baseline 2: Convolutional autoencoder (trained from scratch)
- **Architecture:** encoder of conv + stride/pooling layers down to a small bottleneck, mirrored decoder with upsampling / transposed convolution.
- **Training:** on normal images only, with two loss variants:
  - (a) L2 loss
  - (b) SSIM loss (Bergmann et al. 2019)
- **Anomaly map:** per-pixel L2 residual, or 1 − local SSIM.
- **Why it can fail:** the autoencoder may also reconstruct the defect well, so the residual stays small. This is a known failure mode worth showing.

### 3.3 Main method: PaDiM (Defard et al. 2021)
1. **Patch embedding:**
   - Pass each image (224×224) through a pretrained ResNet-18.
   - Take feature maps from layer1, layer2 and layer3 (64 + 128 + 256 = 448 channels).
   - Bilinearly upsample them to layer1's 56×56 grid and concatenate.
   - Each grid position (i, j) now has a 448-dim vector describing a patch. Its receptive field covers several scales.
2. **Dimension reduction:** keep d = 100 dimensions, chosen either by a random subset (as in the paper) or by PCA (my ablation).
3. **Gaussian per position:** for each (i, j), estimate the mean μᵢⱼ and covariance Σᵢⱼ from the N training images (MLE), with regularization Σᵢⱼ + εI (ε = 0.01) so it is invertible.
4. **Anomaly score per patch:** Mahalanobis distance M(xᵢⱼ) = √((xᵢⱼ − μᵢⱼ)ᵀ Σᵢⱼ⁻¹ (xᵢⱼ − μᵢⱼ)).
5. **Anomaly map:** 56×56 score map → bilinear upsample to 224×224 → Gaussian smoothing (σ = 4).
6. **Image score:** maximum of the anomaly map.
7. **Defect mask:** threshold the map (Otsu, or a threshold from validation normal images).

**Fitting cost:** one forward pass over the training set plus 56×56 covariance estimates, which takes minutes on Colab and needs no network training.

### 3.4 Optional extension (only if time allows): PatchCore (Roth et al. 2022)
Replace the per-position Gaussian with a memory bank of normal patch features and a nearest-neighbour distance, using greedy coreset subsampling to keep the bank small. This gives a "parametric vs non-parametric" comparison: Gaussian vs nearest-neighbour, which echoes the KDE / mean-shift ideas from the Segmentation lecture.

## 4. Experiments and ablations

| Experiment | What is compared | What is measured |
|---|---|---|
| Method comparison | PCA vs autoencoder (L2, SSIM) vs PaDiM | Image AUROC, pixel AUROC, time |
| Distance | Euclidean vs Mahalanobis on the same features | AUROC (the lecture slide's claim, tested on real data) |
| Feature layers | layer1 only, layer1+2, layer1+2+3 | AUROC; effect of receptive-field size |
| Dimension reduction | Random d vs PCA d, for d = 50, 100, 200 | AUROC, memory |
| Covariance regularization | ε from 0.001 to 1 | AUROC, stability |
| Training-set size | 10, 25, 50, 100 %, and few-shot (10 images) | AUROC |
| Robustness | Gaussian noise, blur, brightness/contrast shift at test time; with and without bilateral / CLAHE pre-filtering | AUROC drop |
| Failure analysis | Per defect type: small vs large defects, textures vs objects | Annotated examples |

**Dataset: MVTec AD**
- 5354 high-resolution images, 15 categories (5 textures, 10 objects), more than 70 defect types, with pixel-precise ground-truth masks.
- I will use 5 categories to fit the time limit: carpet, grid, leather (textures) and bottle, hazelnut (objects). Others will be added if time permits.
- **Licence:** CC BY-NC-SA 4.0 (non-commercial use only).

**Metrics:**
- **Image-level AUROC** (detection).
- **Pixel-level AUROC** (localization), using the ground-truth masks.
- **PRO score** (optional): per-region overlap, so large defects do not dominate.
- **Run time** for fitting and per image.
- Qualitative grids: input / ground truth / anomaly map / predicted mask for each method.

## 5. Timeline (2–9 Nov)

| Dates | Work | Done when |
|---|---|---|
| 2–3 Nov | Data loading, evaluation code (AUROC), PCA baseline | PCA AUROC numbers on 5 categories |
| 3–5 Nov | PaDiM pipeline | AUROC close to the paper's R18-Rd100 results |
| 5–6 Nov | Autoencoder (L2 and SSIM), trained on Colab | Both variants evaluated |
| 6–8 Nov | Ablations and robustness study | All tables filled |
| 8–9 Nov | Figures, slides, code clean-up, upload | Presentation ready |

**If time runs short, cut in this order:** PatchCore extension → robustness study → extra categories. The core comparison and the ablations stay.

## 6. Code policy

- I will write all core functions myself: PCA baseline, autoencoder, PaDiM fitting and scoring, evaluation.
- Libraries: PyTorch / torchvision (pretrained ResNet-18 and data loading), NumPy, scikit-learn (only `roc_auc_score`), scikit-image (SSIM, Otsu).
- Public PaDiM implementations exist, e.g. github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master and the anomalib library. I will use them only to check my numbers, cite them, and not copy them.

## 7. References and links

1. T. Defard, A. Setkov, A. Loesch, R. Audigier, "PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization," in *Pattern Recognition. ICPR International Workshops and Challenges*, LNCS vol. 12664, pp. 475–489, Springer, 2021. doi:10.1007/978-3-030-68799-1_35
   - Springer: https://link.springer.com/chapter/10.1007/978-3-030-68799-1_35
   - arXiv: https://arxiv.org/abs/2011.08785
2. P. Bergmann, M. Fauser, D. Sattlegger, C. Steger, "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection," in *Proc. IEEE/CVF CVPR*, pp. 9584–9592, 2019. doi:10.1109/CVPR.2019.00982
   - PDF: https://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf
3. P. Bergmann, K. Batzner, M. Fauser, D. Sattlegger, C. Steger, "The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection," *International Journal of Computer Vision*, vol. 129, no. 4, pp. 1038–1059, 2021. doi:10.1007/s11263-020-01400-4
   - Dataset and evaluation code: https://www.mvtec.com/company/research/datasets/mvtec-ad
4. P. Bergmann, S. Löwe, M. Fauser, D. Sattlegger, C. Steger, "Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders," in *Proc. VISAPP (VISIGRAPP 2019)*, vol. 5, pp. 372–380, 2019. doi:10.5220/0007364503720380
   - arXiv: https://arxiv.org/abs/1807.02011
5. K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE CVPR*, pp. 770–778, 2016. doi:10.1109/CVPR.2016.90
   - PDF: https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
6. (Optional extension) K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, P. Gehler, "Towards Total Recall in Industrial Anomaly Detection," in *Proc. IEEE/CVF CVPR*, pp. 14318–14328, 2022. doi:10.1109/CVPR52688.2022.01392
   - arXiv: https://arxiv.org/abs/2106.08265
   - Code: https://github.com/amazon-research/patchcore-inspection

**Reference code (read and cite, never copy):**
- PaDiM (unofficial PyTorch): https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
- anomalib (Intel, many methods incl. PaDiM and PatchCore): https://github.com/openvinotoolkit/anomalib
