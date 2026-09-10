# VAE vs. DDPM — Generative Models From Scratch (Fashion-MNIST)

**Task code:** GenCV003 &nbsp;|&nbsp; **Position:** CV Engineer

Two generative model families implemented **from scratch** in TensorFlow/Keras, trained on the
same public dataset, and compared quantitatively and qualitatively:

1. **Variational Autoencoder (VAE)** — encoder/decoder pair trained with the reparameterization
   trick and the ELBO (reconstruction + KL) loss.
2. **Denoising Diffusion Probabilistic Model (DDPM)** — a compact U-Net trained to reverse a
   fixed Gaussian noising process, one small step at a time.

Full write-up, architecture details, equations, and results are in
[`GenCV003_VAE_vs_DDPM_Report.pdf`](./GenCV003_VAE_vs_DDPM_Report.pdf). All code lives in the
single notebook [`VAE_vs_DDPM_fashion_mnist.ipynb`](./VAE_vs_DDPM_fashion_mnist.ipynb).

## ✓ Verified

The sample images saved in the notebook (training-data preview, VAE reconstructions/generations,
DDPM generations, and the final real-vs-VAE-vs-DDPM comparison grid) were checked directly and
show clothing items, confirming this is a genuine end-to-end run on **Fashion-MNIST**. Both
models were trained for **100 epochs** in this run. See the report for the full numbers.

## Dataset

[**Fashion-MNIST**](https://github.com/zalandoresearch/fashion-mnist) via
`tf.keras.datasets.fashion_mnist` — 70,000 grayscale 28×28 images of clothing across 10 classes.
For fast CPU training, the notebook takes a random 1,800-image subsample (seed = 42), downsizes
to 16×16 (bilinear), normalizes to `[0, 1]`, and splits 85/15 train/test. The model and training
code are resolution-agnostic, so this is a knob, not a hard limit (see "Scaling up" below).

## Repo layout

```
.
├── VAE_vs_DDPM_fashion_mnist.ipynb   # all code: data, VAE, DDPM, evaluation, plots
├── GenCV003_VAE_vs_DDPM_Report.pdf   # full report (architecture, equations, results, conclusions)
├── report_assets/                    # PNG figures extracted from the notebook, used in the report
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt
```

Requires Python 3.10+. CPU is sufficient for this dataset size.

## Run

```bash
jupyter notebook VAE_vs_DDPM_fashion_mnist.ipynb
```

Then **Kernel → Restart & Run All**. Fashion-MNIST downloads automatically on first run
(internet needed once; it's cached locally afterward, at `~/.keras/datasets/`).

### Expected runtime (CPU)

| Stage | Time |
|---|---|
| VAE training (100 epochs) | ≈ 4–5 min |
| DDPM training (100 epochs) | ≈ 12–14 min |
| DDPM sampling for evaluation (200 sequential steps × ~270 images) | ≈ 1 min |
| **Total notebook run** | **≈ 20 min** |

## What each section of the notebook does

| Section | Contents |
|---|---|
| 0–1 | Imports, seeding, Fashion-MNIST loading/subsampling/resizing |
| 2 | VAE: encoder, decoder, reparameterization trick, ELBO loss, training loop, reconstructions + random samples |
| 3 | DDPM: noise schedule, sinusoidal time embedding, U-Net (`SimpleUNet`), training loop, iterative sampler |
| 4 | Quantitative benchmarking: a small Fashion-MNIST classifier stands in for Inception-v3 (no internet access to pretrained ImageNet weights), giving FID-style and Inception-Score-style metrics calibrated to this dataset |
| 5 | Qualitative side-by-side: real vs. VAE-generated vs. DDPM-generated samples |

The written comparison table and conclusions (VAE vs. DDPM trade-offs) live in the report PDF,
not as a separate notebook section.

## Results at a glance

| Model | FID-style (↓ better) | Inception-Score-style (↑ better) |
|---|---|---|
| Real test set | — | 6.07 |
| VAE | 34.97 | 3.88 |
| DDPM | 69.55 | 3.25 |

The VAE outperformed the DDPM on both metrics in this run — see the report §7 for why this is an
expected outcome at this small scale (subsampled data, single-level U-Net, T=200 steps), and what
would typically close the gap (more data, more U-Net capacity, more diffusion steps).

## Key knobs

| Name | Where | Meaning |
|---|---|---|
| `LATENT_DIM` | §2 | VAE bottleneck size (default 16) |
| `VAE_EPOCHS` | §2 | VAE training epochs (default 100) |
| `T`, `beta_start`, `beta_end` | §3 | DDPM diffusion steps / noise schedule (default 200, 1e-4, 0.02) |
| `DDPM_EPOCHS` | §3 | DDPM training epochs (default 100) |
| `base_ch` | §3 (`SimpleUNet`) | U-Net width (default 48) |

## Scaling up / changing the dataset

Only `load_data()` in Section 1 needs to change — everything downstream (`VAE`, `SimpleUNet`,
training loops, FID/IS evaluation) works unchanged:

- **Full Fashion-MNIST** (60k train / 10k test) instead of the 1,800-image subsample: drop the
  `rng.choice` subsampling line.
- **Higher resolution** (e.g. the native 28×28, or up to 32×32): change the `tf.image.resize`
  target size.
- **A different dataset** (e.g. MNIST, CIFAR-10): swap the loader, keep normalization to `[0, 1]`.
  For color images, increase `base_ch` in `SimpleUNet`, add another down/up level, and consider
  more diffusion steps (`T`) and epochs.

## Evaluation metrics — important caveat

FID and Inception Score are normally computed with an ImageNet-pretrained Inception-v3 network.
That network's weights require an internet download not available in this project's training
environment, so a small CNN classifier trained from scratch on Fashion-MNIST is used as a
drop-in feature extractor / label distribution instead. The resulting numbers have the same
mathematical form and the same "lower/higher is better" interpretation, but are **not comparable
to FID/IS values reported in papers using Inception-v3** — they are only meaningful for the
relative VAE-vs-DDPM comparison inside this notebook.

## License / attribution

Fashion-MNIST is released by Zalando Research under the MIT license.
