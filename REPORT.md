# Generative Models Report: VAE vs. DDPM (GenCV003)

## 1. Overview

This report documents the from-scratch implementation, training, and comparative
evaluation of two generative model families:

- **Variational Autoencoder (VAE)** — Kingma & Welling, 2013
- **Denoising Diffusion Probabilistic Model (DDPM)** — Ho et al., 2020

Both models were implemented directly in PyTorch (no pre-built generative model
libraries such as `diffusers` were used) and trained on the same dataset for a
fair, apples-to-apples comparison.

## 2. Dataset

**Dataset used:** scikit-learn's built-in **Digits** dataset — 1,797 grayscale
8×8 images of handwritten digits (0–9), a well-known public subset derived from
the UCI ML handwritten digits dataset.

**Why this dataset:** the development environment used for this submission had
no internet access to external image-hosting servers, which ruled out
downloading the full MNIST dataset (28×28, 60,000 images) at training time.
The Digits dataset ships offline with scikit-learn and is a legitimate, public,
well-known dataset in the same visual domain (handwritten digits), making it a
practical stand-in that still exercises both architectures meaningfully.

**To reproduce on full MNIST instead:** the codebase includes
`data.load_mnist_torchvision()`, a drop-in replacement that downloads and
loads real MNIST (28×28) via `torchvision`, for anyone running this with
internet access (e.g., locally or on Google Colab). See the README for the
one-line swap.

Preprocessing: pixel values are simply scaled to `[0, 1]`; no augmentation was
applied, since generative models are trained to match the data distribution
as-is.

## 3. Model Architectures

### 3.1 VAE

- **Encoder:** two strided convolutions (8×8 → 4×4 → 2×2) followed by two
  linear heads producing `mu` and `logvar` for a 16-dimensional latent space.
- **Reparameterization trick:** `z = mu + eps * std`, `eps ~ N(0, I)`, making
  sampling differentiable.
- **Decoder:** a linear projection followed by two transposed convolutions
  mirroring the encoder, ending in a sigmoid to output pixel values in `[0,1]`.

### 3.2 DDPM

- **Forward (noising) process:** a fixed linear beta schedule over 200
  timesteps; `x_t` is computed from `x_0` in closed form using the cumulative
  product of `alpha`, avoiding an explicit step-by-step loop during training.
- **Denoising network:** a compact U-Net with:
  - Sinusoidal timestep embeddings (as in Transformers) projected through an
    MLP and injected additively into each conv block.
  - Two downsampling conv blocks, a bottleneck, one upsampling block with a
    skip connection from the first downsampling block.
- **Reverse (sampling) process:** starting from pure Gaussian noise, the model
  iteratively predicts and removes noise for `t = 199 → 0`, following the
  standard DDPM posterior mean/variance formulas.

## 4. Loss Functions

| Model | Loss | Reasoning |
|---|---|---|
| VAE | ELBO = reconstruction loss (binary cross-entropy) + KL divergence to `N(0,I)` | BCE is appropriate since pixel values are treated as independent Bernoulli-like intensities in `[0,1]`; the KL term regularizes the latent space to be sampleable and continuous, which is what lets us generate new images by sampling `z ~ N(0,I)` at inference time. |
| DDPM | Mean-squared error between the true noise `eps` and the network's predicted noise `eps_theta(x_t, t)` | This is the simplified training objective derived in Ho et al. (2020) from the full variational bound; in practice it trains far more stably than the full ELBO-style loss and produces better sample quality. |

## 5. Training Setup

- Both models trained for **60 epochs**, batch size 64, Adam optimizer.
- VAE learning rate: 1e-3. DDPM learning rate: 2e-3.
- DDPM timesteps: 200 (fewer than the original paper's 1000, chosen to keep
  training/sampling time tractable on CPU for this small dataset).
- Both trained on identical data splits and hardware for fairness.

Loss curves (`outputs/vae_loss_curve.png`, `outputs/ddpm_loss_curve.png`) show
smooth, stable convergence for both models with no divergence or collapse.

## 6. Quantitative Evaluation

**Important limitation:** the standard FID (Fréchet Inception Distance) and
Inception Score both require a pretrained Inception-V3 network trained on
ImageNet. The development sandbox had no internet access to download these
pretrained weights. Additionally, Inception-V3 features are tuned for natural
RGB photographs, not small grayscale digit images, and are known to give
unreliable FID estimates on far-out-of-domain data like this.

**Approach used instead:** a small CNN classifier was trained from scratch on
the *real* Digits images only (never on generated images), and used as a
domain-specific feature extractor — a documented substitute for FID when
Inception is unsuitable or unavailable. We report:

- **Digit-FID**: Fréchet distance between the real and generated image feature
  distributions in this classifier's feature space (lower = better).
- **Digit-IS**: An Inception-Score analogue computed from the classifier's
  softmax outputs on generated images (higher = better; rewards both
  confident per-image classification and diversity across classes).

### Results

| Model | Digit-FID ↓ | Digit-IS ↑ |
|---|---|---|
| VAE  | 15.19 | 4.17 |
| DDPM | 8.15  | 5.74 |

*(Exact values will vary slightly between reruns due to random sampling —
see `outputs/metrics.txt` for the values from the run that produced this
report.)*

**Interpretation:** DDPM outperforms the VAE on both metrics — its generated
images are closer in feature-distribution to real digits (lower FID) and
occupy the class space more confidently and diversely (higher IS). This
matches the qualitative visual comparison below and is consistent with
widely reported findings in the generative modeling literature.

*To get literature-standard, Inception-based FID/IS numbers, run
`evaluate.py` with internet access after installing `pytorch-fid` or
`torchmetrics[image]` — see README.*

## 7. Qualitative Evaluation

See `outputs/vae_samples.png` and `outputs/ddpm_samples.png` (8×8 grids of
generated samples).

- **Realism:** DDPM samples have sharper strokes and more distinct digit
  boundaries. VAE samples are noticeably blurrier — a well-known consequence
  of the pixel-wise reconstruction loss, which encourages the model to
  average over plausible outputs rather than commit to sharp detail.
- **Diversity:** Both models produce a reasonable variety of digit shapes and
  styles. DDPM samples show slightly more diverse stroke widths and slants;
  the VAE's samples are somewhat more homogeneous, likely a mild symptom of
  "posterior collapse" tendencies common in small-latent-dimension VAEs.
- **Thematic consistency:** Nearly all VAE and DDPM samples are recognizable
  as digit-like shapes (consistent with the training domain), though at this
  low resolution (8×8) and limited training budget, a fraction of samples
  from both models are ambiguous between adjacent digit classes (e.g. 4/9,
  3/8) — expected given the extremely low pixel resolution of the dataset.

## 8. Key Architectural & Practical Differences

| Aspect | VAE | DDPM |
|---|---|---|
| Generation process | Single forward pass through the decoder | Iterative denoising over many timesteps (200 here, often 1000 in the literature) |
| Training stability | Can suffer posterior collapse; KL term needs balancing | Very stable in practice; simple MSE loss |
| Sample quality | Blurrier, due to pixel-wise reconstruction loss | Sharper, more realistic |
| Sampling speed | Fast (one pass) | Slow (requires a full reverse diffusion loop per sample) |
| Latent space | Explicit, structured, continuous — supports interpolation, encoding of real images | No explicit compact latent code; not directly designed for encoding/interpolation |
| Likelihood | Optimizes a tractable lower bound (ELBO) on log-likelihood | Also connected to a variational bound, but the simplified training objective is not directly a likelihood bound |

## 9. Limitations of This Implementation

1. **Dataset resolution and size**: 8×8 grayscale digits are far simpler than
   typical benchmarks (e.g., 32×32+ CIFAR-10 or full MNIST at 28×28); results
   should be read as a proof of correct implementation and relative
   comparison, not as state-of-the-art generative performance.
2. **No standard FID/Inception Score**: as explained in Section 6, we
   substituted a domain-specific classifier-based metric due to offline
   constraints. This is a reasonable proxy but is not directly comparable to
   FID numbers reported elsewhere in the literature.
3. **Reduced diffusion timesteps** (200 vs. the original paper's 1000) and a
   compact U-Net were used to keep training/sampling tractable on CPU within
   the available compute budget; a full-scale implementation would likely
   further improve DDPM sample quality.
4. **Training budget**: 60 epochs on a small dataset; longer training and
   hyperparameter tuning (e.g., learning-rate schedules, larger networks)
   would likely improve both models further.

## 10. Conclusion

Both generative models were successfully implemented from scratch and trained
to convergence on the same dataset. The DDPM consistently outperformed the VAE
on both quantitative metrics (Digit-FID, Digit-IS) and in visual sample
quality, at the cost of a substantially slower, iterative sampling process.
This tradeoff — VAE's speed and structured latent space vs. DDPM's superior
sample fidelity — mirrors the broader tradeoffs reported for these model
families in the generative modeling literature.
