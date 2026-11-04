# MSAR-CassavaNet: Multi-Scale Attention Residual CNN for Cassava Leaf Disease Classification

Official repository for the paper: **"Cassava Leaf Disease Classification with a 1M-Parameter Multi-Scale Attention Residual CNN: A Leakage-Controlled Evaluation"**.

## Abstract
Lightweight convolutional networks are widely proposed for cassava leaf disease diagnosis, but reported results are hard to compare due to differing datasets, split strategies, and evaluation sets. This study evaluates **MSAR-CassavaNet** (1.01M parameters, 147M FLOPs)—a compact architecture composing depthwise-separable residual blocks, multi-scale feature fusion, sparse channel/spatial attention, and focal-loss training.

Under a strict, leakage-controlled protocol on the 21,397-image Cassava Leaf Disease dataset (70/15/15 stratified split, 87 total runs, exact McNemar tests with Holm correction, paired bootstrap CIs, and calibration analysis), MSAR-CassavaNet achieves **82.03 ± 0.70% test accuracy** and **68.48 ± 1.17% macro-F1**.

## Key Findings
- **Architecture Composition:** The full MSAR composition does not statistically outperform a plain separable-residual base network ($69.07 \pm 0.74\%$ macro-F1).
- **Pretraining Advantage:** ImageNet-pretrained baselines fine-tuned for 15 epochs consistently outperform scratch-trained lightweight designs ($83.46\%$ to $87.24\%$ test accuracy).
- **Imbalance & Calibration:** Focal loss ($\gamma=2$) induces strong underconfidence ($	ext{ECE} pprox 0.14$), whereas standard cross-entropy achieves optimal calibration ($	ext{ECE} pprox 0.02$).

---

## Repository Structure & Script Map

| Script | Purpose |
| :--- | :--- |
| `models/msar_cassavanet.py` | PyTorch implementation of MSAR-CassavaNet & ablation modules |
| `run0_calibrate.py` | Expected Calibration Error (ECE), Negative Log-Likelihood & reliability diagrams |
| `run1_main_ablation.py` | Main training and 9-variant component/loss ablation study |
| `run2_baselines.py` | Training & benchmark script for 12 baseline models (scratch & fine-tuned) |
| `run3_cv_robust_cam.py` | 5-Fold Stratified Cross-Validation & Grad-CAM visual interpretability |
| `run4_analysis.py` | Exact McNemar tests (Holm-corrected) & paired bootstrap 95% CIs |

---

## Installation

```bash
git clone https://github.com/your-username/MSAR-CassavaNet.git
cd MSAR-CassavaNet
pip install -r requirements.txt
```

## Dataset Setup

Download the Kaggle Cassava Leaf Disease Classification dataset:
```bash
kaggle competitions download -c cassava-leaf-disease-classification
unzip cassava-leaf-disease-classification.zip -d data/
```

Organize data as:
```
data/
├── train_images/
└── train.csv
```

## Usage

### 1. Run Main Model & Ablation Study
```bash
python run1_main_ablation.py --data_dir data/ --seeds 11 22 33 44 55
```

### 2. Benchmarking Baselines
```bash
python run2_baselines.py --data_dir data/ --epochs_scratch 40 --epochs_pretrained 15
```

### 3. Model Calibration & Reliability Analysis
```bash
python run0_calibrate.py --checkpoint checkpoints/msar_full_seed11.pth
```

### 4. 5-Fold Cross-Validation & Grad-CAM Visualizations
```bash
python run3_cv_robust_cam.py --data_dir data/ --cam_output outputs/gradcam/
```

### 5. Statistical Hypothesis Testing
```bash
python run4_analysis.py --preds_dir results/preds/
```

---

## Model Architecture Details

| Stage | Output Size | Channels | Operations |
| :--- | :---: | :---: | :--- |
| **Stem** | $112 	imes 112$ | 24 | $3 	imes 3$ Conv, stride 2 |
| **Early Down** | $56 	imes 56$ | 32 | Depthwise-Separable Conv, stride 2 |
| **Stage A** | $28 	imes 28$ | 64 | 2 Residual Blocks (Exp. 3); SE in 2nd |
| **Stage B** | $14 	imes 14$ | 128 | Multi-Scale Fusion Block (Skip Connection) |
| **Stage C** | $14 	imes 14$ | 128 | 2 Residual Blocks (Exp. 4); SE in 2nd; Spatial Gate |
| **Stage D** | $7 	imes 7$ | 256 | 1 Residual Block (Exp. 3) with SE |
| **Head** | $1 	imes 1$ | 512 | $1 	imes 1$ Conv, Global Avg Pooling, Dropout(0.2), Linear(5) |

---

## Citation

```bibtex
@article{meva2026msar,
  title={Cassava Leaf Disease Classification with a 1M-Parameter Multi-Scale Attention Residual CNN: A Leakage-Controlled Evaluation},
  author={Meva, Divyakant and Popat, Kalpesh and Kukadiya, Hiren},
  journal={Draft Version 2.3},
  year={2026}
}
```
