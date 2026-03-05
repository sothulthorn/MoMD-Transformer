# MoMD Transformer

A PyTorch reimplementation of the **MoMD Transformer** (Mixture-of-Modality-Diagnosis Transformer) for adaptive multi-modal fault diagnosis via knowledge transfer with vibration-current signals.

> **Reference:** "MoMD Transformer: adaptive multi-modal fault diagnosis via knowledge transfer with vibration-current signals" (Information Fusion, 2026)
>
> **Disclaimer:** This is an independent reimplementation for demonstration purposes only. It is not affiliated with the original authors.

## Overview

The MoMD Transformer performs fault diagnosis using paired vibration and current signals. Its key features:

- **Multi-channel FFN** within each transformer block: separate pathways (V-FFN, C-FFN, MM-FFN) allow the shared attention to specialize per modality.
- **Global Knowledge Transfer (GKT):** aligns vibration-only and current-only class embeddings so each single-modality pathway learns from the other.
- **Masked Signal Modeling (MSM):** a self-supervised auxiliary task that randomly masks input patches and reconstructs them, improving feature robustness.
- **Adaptive inference:** at test time, the model can diagnose faults from vibration only, current only, or both modalities.

## Architecture

![Architecture](assets/MoMD%20Transformer.png)

```
Input: (B, 2048) per modality
         |
    Patchify: (B, 32, 64)
         |
    Linear projection + position/type embeddings: (B, 33, 128)
         |
    3x MoMDBlock:
      - Shared Multi-Head Self-Attention (8 heads)
      - Multi-channel FFN (V-FFN / C-FFN / MM-FFN selected by modality)
         |
    CLS token -> Linear(128, num_classes) -> logits

Multi-modal: [cls_vib, v_1..v_32, cls_cur, c_1..c_32] -> (B, 66, 128)
```

**Parameters:** ~1.2M

## Training Procedure

Each training step performs four forward passes per batch (Section 3.5):

1. Vibration only -> classification loss L_V + block-level CLS features
2. Current only -> classification loss L_C + block-level CLS features
3. Both modalities (no mask) -> classification loss L_VC
4. Both modalities (with mask) -> MSM reconstruction loss

Total loss (Eq. 21):

```
L_all = L_D + lambda_gkt * L_gkt + lambda_msm * L_msm

where L_D = (L_V + L_C + L_VC) / 3
```

## Datasets

### PU Bearing Dataset (Paderborn University)

| Class | Bearings | Samples |
|-------|----------|---------|
| Normal | K001, K002, K003 | 800 each |
| Outer Ring Damage | KA04, KA15, KA16 | 800 each |
| Inner Ring Damage | KI04, KI14, KI16 | 800 each |

- Source: [Paderborn Bearing DataCenter](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter)
- Sampling rate: 64 kHz
- Channels: vibration acceleration + motor phase current

### PMSM Stator Fault Dataset

| Class | Conditions | Samples |
|-------|-----------|---------|
| Normal | 1000W baseline | 1200 |
| Inter-turn Fault | 6.48%, 21.69% severity | 1200 |
| Inter-coil Fault | 2.00%, 7.56% severity | 1200 |

- Source: [Mendeley Data](https://data.mendeley.com/datasets/rgn5brrgrn/5)
- Vibration: 25.6 kHz, Current: 100 kHz (downsampled to 25.6 kHz)

## Project Structure

```
MoMD-Transformer/
  config.py        - Hyperparameters and dataset configurations
  model.py         - MoMD Transformer architecture (MoMDBlock, multi-channel FFN, GKT, MSM)
  dataset.py       - Dataset loading with stratified train/val/test split
  train.py         - Training loop, evaluation, and multi-run aggregation
  preprocess.py    - Raw data conversion (.mat/.tdms -> .npy)
  utils.py         - Evaluation, confusion matrices, t-SNE visualization
  requirements.txt - Python dependencies
  data/
    pu/            - Preprocessed PU dataset (.npy files)
    pmsm/          - Preprocessed PMSM dataset (.npy files)
    pu_raw/        - Raw PU .mat files
    pmsm_raw/      - Raw PMSM .tdms files
  results/         - Training outputs (models, plots, CSVs)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Preprocessing

Download the raw datasets to `data/pu_raw/` and `data/pmsm_raw/`, then:

```bash
# Inspect raw file structure
python preprocess.py --dataset pu --raw_dir ./data/pu_raw --inspect

# Preprocess
python preprocess.py --dataset pu   --raw_dir ./data/pu_raw   --output_dir ./data/pu
python preprocess.py --dataset pmsm --raw_dir ./data/pmsm_raw --output_dir ./data/pmsm

# Optional: skip normalization
python preprocess.py --dataset pu --raw_dir ./data/pu_raw --output_dir ./data/pu_nonorm --norm none
```

## Usage

### Training

```bash
# Train on PU dataset (10 repeats, 200 epochs)
python train.py --dataset pu

# Train on PMSM dataset with custom settings
python train.py --dataset pmsm --epochs 200 --lr 1e-4 --batch_size 64 --repeats 10
```

### Outputs

Each run produces:
- `training_history.csv` — per-epoch losses and accuracies
- `training_curve.png` — loss and accuracy plots
- `confusion_matrix.png` — per-modality confusion matrices
- `tsne.png` — t-SNE feature visualizations
- `model.pt` — best model checkpoint

After all repeats, a `summary.csv` reports mean and std across runs.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Signal length | 2048 |
| Segment (patch) length | 64 |
| Embedding dim | 128 |
| MLP dim | 512 |
| Attention heads | 8 |
| Encoder depth | 3 |
| Dropout | 0.2 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 64 |
| Epochs | 200 |
| Mask ratio (MSM) | 0.15 |
| Lambda GKT | 1.0 |
| Lambda MSM | 1.0 |
| Train/Val/Test split | 60/20/20 |

## Requirements

- Python 3.10+
- PyTorch
- NumPy, SciPy, scikit-learn
- matplotlib, tqdm
- nptdms (for PMSM raw data)
