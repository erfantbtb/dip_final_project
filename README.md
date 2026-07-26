# MERIT — RAW-to-RAW Image Translation

A simplified PyTorch implementation of **MERIT** for unpaired RAW-to-RAW image translation between different camera sensor domains. The model translates RAW images from one camera domain into the style of another, using only unpaired data during training.

## Overview

- **Style Encoder** extracts domain/style information from a reference RAW image.
- **Generator** (U-Net with **MSLKA** — Multi-Scale Large Kernel Attention) performs the translation conditioned on the style code.
- Trained with a **cycle-consistency loss** and a **style-reconstruction loss**, no adversarial discriminator needed.
- A **brightness-based reference selection** picks the best style exemplar at inference time.

## Project Structure

```
src/
  model.py              — StyleEncoder, MSLKA, Generator, MERIT
  data_generator.py     — Synthetic domain profiles & data generators
  data_preprocessing.py — RAW linearization & Bayer pack/unpack
  train.py              — Training loop
  evaluate.py           — Metrics (MAE, PSNR, SSIM, KL) & visualization
  main.py               — End-to-end demo: train, translate, evaluate
MERIT.pdf               — Paper reference
requirements.txt        — Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py
```

This runs a full demo:
1. Creates two synthetic camera domains (`CameraA`, `CameraB`) with different tint/noise profiles.
2. Trains MERIT on unpaired samples from both domains for 150 steps.
3. Translates a held-out scene from Domain A into Domain B's style.
4. Evaluates against paired ground truth with MAE / PSNR / SSIM / KL divergence.
5. Saves a side-by-side comparison figure to `translation_result.png`.

## Architecture (simplified)

- **StyleEncoder**: 4-layer conv → adaptive pooling → MLP → 64-dim style vector.
- **Generator**: Encoder-decoder with skip connections. Each up-block contains an **MSLKA** module that applies multi-dilation depthwise convolutions fused with a style-modulated channel gate.
- **Loss**: `L = λ_cycle * L1(cycle) + λ_style * L1(style_recon)`.

## Dependencies

- PyTorch ≥ 2.0
- Matplotlib ≥ 3.7
- NumPy ≥ 1.24
