
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg

from data import DigitsDataset
from vae import VAE
from ddpm import SimpleUNet, DiffusionSchedule, sample_ddpm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 8
LATENT_DIM = 16
TIMESTEPS = 200


class TinyClassifier(nn.Module):
    def __init__(self, n_classes=10, feat_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.fc_feat = nn.Linear(32 * 2 * 2, feat_dim)
        self.fc_out = nn.Linear(feat_dim, n_classes)

    def features(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.pool(h)
        h = h.view(h.size(0), -1)
        return F.relu(self.fc_feat(h))

    def forward(self, x):
        feat = self.features(x)
        return self.fc_out(feat)


def train_classifier(epochs=20):
    dataset = DigitsDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    clf = TinyClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            logits = clf(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return clf


def get_features_and_probs(clf, images):
    clf.eval()
    with torch.no_grad():
        feats = clf.features(images).cpu().numpy()
        probs = F.softmax(clf(images), dim=1).cpu().numpy()
    return feats, probs


def frechet_distance(feats_real, feats_fake, eps=1e-6):
    mu1, sigma1 = feats_real.mean(axis=0), np.cov(feats_real, rowvar=False)
    mu2, sigma2 = feats_fake.mean(axis=0), np.cov(feats_fake, rowvar=False)

  
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def inception_score_like(probs, eps=1e-10):
    p_y = probs.mean(axis=0, keepdims=True)
    kl = probs * (np.log(probs + eps) - np.log(p_y + eps))
    kl = kl.sum(axis=1)
    return float(np.exp(kl.mean()))


def main():
    print("Training domain-specific feature-extractor classifier on real digits...")
    clf = train_classifier(epochs=20)

    real_dataset = DigitsDataset()
    real_images = real_dataset.images.to(DEVICE)
    feats_real, _ = get_features_and_probs(clf, real_images)

    results = {}

    #  VAE 
    vae = VAE(in_channels=1, latent_dim=LATENT_DIM, base_ch=32).to(DEVICE)
    vae.load_state_dict(torch.load("outputs/vae.pt", map_location=DEVICE))
    vae.eval()
    vae_samples = vae.sample(len(real_dataset), DEVICE, LATENT_DIM)
    feats_vae, probs_vae = get_features_and_probs(clf, vae_samples)
    results["VAE"] = {
        "Digit-FID": frechet_distance(feats_real, feats_vae),
        "Digit-IS": inception_score_like(probs_vae),
    }

    #  DDPM 
    ddpm = SimpleUNet(in_channels=1, base_ch=32, time_dim=64).to(DEVICE)
    ddpm.load_state_dict(torch.load("outputs/ddpm.pt", map_location=DEVICE))
    schedule = DiffusionSchedule(timesteps=TIMESTEPS, device=DEVICE)
    print("Sampling DDPM for evaluation (this loops over all diffusion steps)...")
    ddpm_samples = sample_ddpm(ddpm, schedule, n_samples=len(real_dataset),
                                img_size=IMG_SIZE, device=DEVICE)
    feats_ddpm, probs_ddpm = get_features_and_probs(clf, ddpm_samples)
    results["DDPM"] = {
        "Digit-FID": frechet_distance(feats_real, feats_ddpm),
        "Digit-IS": inception_score_like(probs_ddpm),
    }

    print("\n===== Quantitative Results (lower FID = better, higher IS = better) =====")
    for model_name, metrics in results.items():
        print(f"{model_name:6s} | Digit-FID: {metrics['Digit-FID']:8.3f} | "
              f"Digit-IS: {metrics['Digit-IS']:.3f}")

    with open("outputs/metrics.txt", "w") as f:
        f.write("Quantitative Results (lower FID = better, higher IS = better)\n")
        f.write("=" * 60 + "\n")
        for model_name, metrics in results.items():
            f.write(f"{model_name:6s} | Digit-FID: {metrics['Digit-FID']:8.3f} | "
                    f"Digit-IS: {metrics['Digit-IS']:.3f}\n")

    return results


if __name__ == "__main__":
    main()
