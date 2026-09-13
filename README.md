# VAE vs. DDPM — From-Scratch Generative Models 

Implementation and comparison of a Variational Autoencoder (VAE) and a
Denoising Diffusion Probabilistic Model (DDPM), both built from scratch in
PyTorch, trained on scikit-learn's Digits dataset.

See **REPORT.md** for the full write-up: architecture details, loss function
reasoning, quantitative + qualitative results, and limitations.

## Project structure

```
.
├── vae.py            # VAE model (encoder, decoder, reparameterization, loss)
├── ddpm.py           # DDPM model (noise schedule, U-Net, forward/reverse process)
├── data.py           # Dataset loading (Digits dataset by default, MNIST optional)
├── train_vae.py       # Trains the VAE, saves model + samples + loss curve
├── train_ddpm.py      # Trains the DDPM, saves model + samples + loss curve
├── evaluate.py        # Quantitative comparison (Digit-FID, Digit-IS)
├── requirements.txt
├── REPORT.md
└── outputs/           # Generated after running the scripts (not committed)
    ├── vae.pt, ddpm.pt
    ├── vae_samples.png, ddpm_samples.png
    ├── vae_loss_curve.png, ddpm_loss_curve.png
    └── metrics.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the results

```bash
# 1. Train the VAE (~1-2 minutes on CPU)
python3 train_vae.py

# 2. Train the DDPM (~5-10 minutes on CPU, sampling loops over 200 timesteps)
python3 train_ddpm.py

# 3. Run quantitative + qualitative evaluation
python3 evaluate.py
```

All outputs (trained model weights, sample-image grids, loss curves, and
`metrics.txt`) are written to `outputs/`.

## Using full MNIST instead of the Digits dataset

This project defaults to scikit-learn's Digits dataset (8×8 images) for fast,
fully self-contained iteration with no external downloads required. To
reproduce on full 28×28 MNIST instead:

1. In `train_vae.py` and `train_ddpm.py`, replace:
   ```python
   from data import get_dataloader
   loader = get_dataloader(batch_size=BATCH_SIZE)
   ```
   with:
   ```python
   from data import load_mnist_torchvision
   loader = load_mnist_torchvision(batch_size=BATCH_SIZE)
   ```
2. Update `IMG_SIZE = 28` in both training scripts (and in `evaluate.py`).
3. Re-run the steps above. Note: DDPM sampling time scales with image size
   and will take noticeably longer at 28×28 than at 8×8.

## Getting literature-standard FID / Inception Score

The `evaluate.py` script uses a domain-specific classifier-based metric
(explained in REPORT.md), since standard Inception-V3 features are tuned for
natural RGB photographs and give unreliable estimates on small grayscale
digit images. To additionally compute standard, literature-comparable FID/IS:

```bash
pip install pytorch-fid torchmetrics[image]
```

then use `torchmetrics.image.fid.FrechetInceptionDistance` and
`torchmetrics.image.inception.InceptionScore` on the generated sample tensors
in `outputs/` in place of the custom functions in `evaluate.py`.

## Notes

- Both models are intentionally lightweight so the whole pipeline (train +
  sample + evaluate) runs on CPU in a few minutes. For higher-fidelity
  results, increase `base_ch`, `EPOCHS`, and (for DDPM) `TIMESTEPS` in the
  training scripts.
- Random seeds are not fixed by default; set `torch.manual_seed(...)` at the
  top of the training scripts for exact reproducibility across runs.
