# Cassava leaf disease: a controlled, multi-seed benchmark of imbalance- and noise-aware training for lightweight classifiers

Code, split file and saved results for the study *"Imbalance- and Noise-Aware Training for Lightweight Cassava Leaf Disease Classification: A Controlled Multi-Seed Benchmark on Field Images"* (manuscript in preparation).

The study asks whether three components help a lightweight MobileNetV3-Large classifier on 21,397 cassava field images:
**A**, a small local–global attention head called LANet; **L**, logit-adjusted loss; and **N**, a class-aware sample re-weighting module for noisy labels.
All 8 on/off combinations (a 2³ factorial design) were trained with 3 seeds and compared with five standard architectures, two further imbalance losses and a controlled label-noise experiment.

> **Headline finding (negative).** No combination of the three components improved test macro-F1 over a plain global-average-pooling head (73.1 ± 0.3 %). Logit adjustment raised CBB recall but lowered CBB precision and macro-F1. Under injected label noise, symmetric cross-entropy was the most robust method; the re-weighting module did not help. Only larger models (ConvNeXt-Tiny, MobileViT-S) improved macro-F1, at much higher cost. The results are reported in full below.

---

## Repository layout

```
kaggle_scripts/                 the three self-contained scripts that produced all results (run on Kaggle, T4 GPU)
  cassava_run1_ablation.py          Run 1: 8-configuration ablation x 3 seeds (24 training runs)
  cassava_run2_baselines_losses.py  Run 2: 5 baseline architectures + focal / class-balanced loss x 3 seeds (21 runs)
  cassava_run3_noise_analysis.py    Run 3: label-noise experiment (24 runs) + efficiency table (+ robustness/explainability code, see below)
analysis/
  make_tables_and_figures.py        regenerates every table / figure from the saved results (no GPU needed)
  draw_architecture.py              redraws the LANet diagram
data/
  splits.csv                        the exact train / val / test assignment (image_name, split, group) -- no labels, no images
results/
  run1_ablation/  run2_baselines_losses/  run3_noise_efficiency/
      results.jsonl                 one JSON record per training run (config, seed, metrics, timing, library versions)
      preds/*.npy                   test-set predictions of every run (int8, test images in splits.csv order)
      table_*.csv                   tables written by the scripts
  summary/                          JSON/CSV produced by analysis/make_tables_and_figures.py
figures/                            figures used in the paper
```

## Data

The images are **not** included and are not redistributed here.

* **Source used.** The Kaggle dataset `nirmalsankalana/cassava-leaf-disease-classification`
  (mounted in the runs at `/kaggle/input/datasets/nirmalsankalana/cassava-leaf-disease-classification/data/`; page URL reconstructed from that path:
  https://www.kaggle.com/datasets/nirmalsankalana/cassava-leaf-disease-classification — please confirm before citing). It stores the images in one folder per class.
  The file names are numeric identifiers, and the total (21,397 images) and the class counts match the training set of the Kaggle competition *Cassava Leaf Disease Classification*
  (https://www.kaggle.com/competitions/cassava-leaf-disease-classification). **We did not compare the labels image by image with the competition's `train.csv`.**
* **Folder → class mapping** used by the scripts (by keyword):

| Folder | Class index | Class | Images |
| --- | ---: | --- | ---: |
| `Cassava___bacterial_blight` | 0 | CBB | 1,087 |
| `Cassava___brown_streak_disease` | 1 | CBSD | 2,189 |
| `Cassava___green_mottle` | 2 | CGM | 2,386 |
| `Cassava___mosaic_disease` | 3 | CMD | 13,158 |
| `Cassava___healthy` | 4 | Healthy | 2,577 |

* **Split.** Near-duplicate images (64-bit difference hash, Hamming distance ≤ 2) are grouped (no such groups were found in this data), then a stratified group split
  (`StratifiedGroupKFold`, 10 folds, seed 42) gives 17,117 train / 2,140 validation / 2,140 test images. The same split was used in every run (split hash `c622c4b8b2` is stored in each record).
  `data/splits.csv` lists the assignment by file name. It contains no labels; check the dataset's terms of use before adding them.
* The scripts also accept the original competition layout (`train.csv` + `train_images/`).

## Environment used

Kaggle notebook, one Tesla T4 GPU, Python 3.13, PyTorch 2.11.0 (CUDA 12.8), timm 1.0.29 (versions are stored in every `results.jsonl` record). See `requirements.txt`.
Training uses cuDNN benchmark mode and mixed precision, so re-running a seed can give slightly different numbers.

## Reproducing

### A. Regenerate the tables and figures (CPU, seconds)

```bash
pip install -r requirements.txt
python analysis/make_tables_and_figures.py --data-dir /path/to/your/copy/of/the/cassava/data
```

`--data-dir` is only used to read the test labels (file names and folder names; no image is opened). The script first checks that the stored predictions reproduce the recorded macro-F1, which also confirms that your labels match the ones used here.

### B. Retrain on Kaggle

1. New Kaggle notebook, accelerator **GPU T4**, Internet **on** (pretrained timm weights are downloaded), attach the Cassava data.
2. Paste a script from `kaggle_scripts/` into a cell. First run it with `SMOKE = True` (about 5 minutes, 1,500 images), then with `SMOKE = False` and *Save & Run All*.
3. Run 1, 2 and 3 are independent; each prints its tables at the end. Every script skips runs that are already recorded in an attached `results.jsonl`, and stops starting new runs after 10.5 h of a 12 h session.
4. Measured training time on the T4: Run 1 about 3.3 h, Run 2 about 4.0 h, Run 3 about 2.9 h in total.

## Method summary

* **Backbone** (all variants): timm `mobilenetv3_large_100`, ImageNet-pretrained, stride-16 and stride-32 feature maps.
* **Base**: global average pooling of the stride-32 map + dropout 0.2 + linear (2.98 M parameters, 0.21 GMACs).
* **LANet (A)**: 1×1 projections to 128 channels; depthwise 3×3 on the stride-16 branch gated by the upsampled stride-32 branch; a 14×14 attention map used for attention pooling;
  sparsity penalty 0.05·mean(A) (3.13 M parameters, 0.23 GMACs). See `figures/architecture.png`.
* **L**: cross-entropy of logits + log class prior (τ = 1).
* **N**: per-class quantile (0.8) of recent losses sets a threshold; samples above it get weight exp(−(loss − threshold)·s/T); for LANet the factor s uses the entropy of the attention map; 3 warm-up epochs.
* **Training** (all models): AdamW (weight decay 0.02), batch 64, ≤ 35 epochs, 200-step warm-up + cosine decay, fp16, GPU augmentation (random resized crop, flip, ±15° rotation, brightness/contrast/saturation), early stopping (patience 8) on validation macro-F1; the best-validation weights are evaluated once on the test set.
  Learning rates were set from common practice and **not tuned** per model.

## Results (test set, 2,140 images)

Mean ± SD over 3 seeds unless stated; percentages. Δ = seed-averaged macro-F1 difference to the base model with a 95 % bootstrap interval over test images (2,000 resamples).

### Factorial ablation (Run 1)

| Config | Accuracy | Macro-F1 | Balanced acc. | CBB recall | Δ macro-F1 vs base [95% CI] | Params (M) | GMACs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base (GAP head) | 84.5 ± 0.2 | 73.1 ± 0.3 | 72.7 ± 0.7 | 59.4 ± 0.5 | — | 2.98 | 0.21 |
| A (LANet head) | 84.5 ± 0.2 | 73.0 ± 0.3 | 72.3 ± 0.3 | 61.5 ± 6.2 | −0.09 [−1.44, +1.23] | 3.13 | 0.23 |
| L (logit-adjusted loss) | 82.5 ± 0.9 | 71.0 ± 0.8 | 73.1 ± 0.5 | 65.8 ± 4.3 | −2.12 [−3.60, −0.76] | 2.98 | 0.21 |
| N (noise module) | 83.6 ± 1.4 | 71.9 ± 1.9 | 71.9 ± 1.6 | 61.2 ± 2.8 | −1.19 [−2.50, +0.09] | 2.98 | 0.21 |
| A+L | 82.8 ± 0.6 | 71.1 ± 0.2 | 73.0 ± 1.7 | 65.2 ± 12.3 | −1.99 [−3.53, −0.44] | 3.13 | 0.23 |
| A+N | 83.5 ± 0.9 | 71.6 ± 0.2 | 71.6 ± 1.1 | 56.4 ± 5.1 | −1.48 [−2.85, −0.18] | 3.13 | 0.23 |
| L+N | 84.4 ± 0.9 | 72.5 ± 0.8 | 72.2 ± 0.1 | 53.3 ± 3.7 | −0.58 [−2.02, +0.80] | 2.98 | 0.21 |
| A+L+N (full) | 84.0 ± 1.3 | 72.5 ± 0.6 | 72.8 ± 1.2 | 58.2 ± 9.2 | −0.61 [−2.03, +0.78] | 3.13 | 0.23 |

### Baselines (Run 2)

| Model | Accuracy | Macro-F1 | Balanced acc. | CBB recall | Δ macro-F1 vs base [95% CI] | Params (M) | GMACs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base (MobileNetV3-L backbone, GAP head) | 84.5 ± 0.2 | 73.1 ± 0.3 | 72.7 ± 0.7 | 59.4 ± 0.5 | — | 2.98 | 0.21 |
| MobileNetV3-L (standard head) | 82.6 ± 0.8 | 68.8 ± 1.4 | 67.8 ± 1.9 | 45.2 ± 5.3 | −4.32 [−5.89, −2.75] | 4.21 | 0.22 |
| GhostNet-100 | 84.0 ± 0.3 | 71.6 ± 0.8 | 70.4 ± 1.4 | 51.2 ± 8.2 | −1.54 [−3.16, +0.04] | 3.91 | 0.14 |
| EfficientNet-B0 | 84.5 ± 0.3 | 72.8 ± 0.9 | 71.5 ± 1.7 | 59.7 ± 3.8 | −0.29 [−1.73, +1.09] | 4.01 | 0.38 |
| MobileViT-S | 85.8 ± 0.1 | 74.5 ± 0.1 | 73.4 ± 0.7 | 58.8 ± 3.7 | +1.45 [−0.09, +2.99] | 4.94 | 1.52 |
| ConvNeXt-Tiny | 86.4 ± 0.8 | 75.6 ± 1.6 | 75.1 ± 1.8 | 67.6 ± 11.7 | +2.51 [+0.80, +4.02] | 27.82 | 4.45 |

### Imbalance losses on the base model (Runs 1–2)

| Loss (base model) | Accuracy | Macro-F1 | Balanced acc. | CBB recall |
| --- | ---: | ---: | ---: | ---: |
| Cross-entropy | 84.5 ± 0.2 | 73.1 ± 0.3 | 72.7 ± 0.7 | 59.4 ± 0.5 |
| Logit-adjusted | 82.5 ± 0.9 | 71.0 ± 0.8 | 73.1 ± 0.5 | 65.8 ± 4.3 |
| Focal (γ=2) | 84.1 ± 0.3 | 71.8 ± 0.8 | 71.0 ± 0.9 | 57.9 ± 1.9 |
| Class-balanced | 84.3 ± 0.3 | 72.1 ± 0.5 | 71.8 ± 1.5 | 59.4 ± 6.8 |

### Label noise (Run 3; 2 seeds)

Symmetric noise (labels replaced by a random other class with probability 10/20/30 %) in the **training** labels only; validation and test labels are clean. The SD of two seeds is only a crude indication. The clean column re-uses the Run 1 results for the same two seeds; symmetric cross-entropy was not run on clean labels.

| Method | Clean (0%) | 10% noise | 20% noise | 30% noise |
| --- | ---: | ---: | ---: | ---: |
| Base + CE | 72.9 ± 0.2 | 72.6 ± 0.3 | 68.2 ± 1.6 | 66.8 ± 0.7 |
| A+L | 71.0 ± 0.1 | 70.2 ± 1.2 | 67.2 ± 0.1 | 64.1 ± 1.5 |
| A+L+N (full) | 72.8 ± 0.2 | 68.4 ± 0.7 | 67.7 ± 1.0 | 64.2 ± 0.5 |
| Base + symmetric CE | not run | 72.3 ± 0.2 | 70.9 ± 2.0 | 70.7 ± 2.4 |

Paired differences to Base + CE (seed-averaged, 95 % bootstrap interval):

| Comparison (macro-F1) | 10% noise | 20% noise | 30% noise |
| --- | ---: | ---: | ---: |
| A+L vs CE | −2.39 [−3.93, −0.77] | −0.99 [−2.63, +0.66] | −2.70 [−4.65, −0.82] |
| Full vs CE | −4.22 [−5.90, −2.52] | −0.51 [−2.40, +1.33] | −2.63 [−4.68, −0.55] |
| SCE vs CE | −0.33 [−2.17, +1.50] | +2.75 [+0.71, +4.74] | +3.87 [+1.93, +5.81] |
| Full vs A+L (effect of the re-weighting module) | −1.83 [−3.51, −0.21] | +0.48 [−1.11, +2.04] | +0.07 [−1.39, +1.50] |

Mean sample weight given by the re-weighting module (full model, after warm-up):

| Noise | Mean weight, clean labels | Mean weight, flipped labels |
| --- | ---: | ---: |
| 10% | 0.92 | 0.32 |
| 20% | 0.93 | 0.58 |
| 30% | 0.94 | 0.71 |

### Cost (Run 3; single measurement on the Kaggle machine)

| Model | Params (M) | GMACs | CPU latency, bs 1 (ms) | GPU latency, bs 1 (ms) | GPU throughput, bs 64, fp16 (img/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| LANet (A) | 3.13 | 0.23 | 25.6 | 7.1 | 2449 |
| MobileNetV3-L (standard head) | 4.21 | 0.22 | 24.0 | 6.5 | 2741 |
| GhostNet-100 | 3.91 | 0.14 | 33.5 | 10.3 | 2154 |
| EfficientNet-B0 | 4.01 | 0.38 | 33.5 | 8.2 | 1622 |
| MobileViT-S | 4.94 | 1.52 | 54.6 | 9.1 | 667 |
| ConvNeXt-Tiny | 27.82 | 4.45 | 76.7 | 7.4 | 664 |

## What was *not* run

* `kaggle_scripts/cassava_run3_noise_analysis.py` also contains code for corruption robustness and attention-map faithfulness (Grad-CAM, deletion/insertion). **These analyses were not executed on real checkpoints** (the Run 1 checkpoints were not available to Run 3) and **no results from them are reported**. The code path was only exercised on a small synthetic dataset. Treat it as untested.
* Model checkpoints are not included.
* No hyper-parameter tuning, no resolution above 224×224, no other datasets.

## Limitations

Single split; 3 seeds (2 for the noise experiment); untuned baselines (the standard MobileNetV3-Large head scored 4.3 points below the base head on the same backbone, which we did not investigate);
the test set has only 110 CBB images, so CBB recall differences of a few points are not resolvable; the noise experiment uses symmetric noise and a clean validation set;
latency was measured once on a shared cloud machine and not on mobile hardware.

## `results.jsonl` fields

`run_id` (`abl_<config>__s<seed>`, `bl_<model>__s<seed>`, `loss_<name>__s<seed>`, `noise<rate>_<method>__s<seed>`), `seed`, `spec`, `loss`, `noise_module`, `noise_rate`, `noise_actual`,
`params_m`, `gmacs`, `best_epoch`, `epochs_run`, `val_macro_f1`, `test` (accuracy, macro-F1, balanced accuracy, per-class precision/recall/F1), `train_s`, `split_hash`, `nz_w_clean`, `nz_w_noisy`, library versions and GPU name.

## License and citation

Code: MIT License (see `LICENSE`). The Cassava images and labels are subject to their own terms on Kaggle and are not part of this repository.
Please cite the paper when it is available; `CITATION.cff` contains a provisional entry.
