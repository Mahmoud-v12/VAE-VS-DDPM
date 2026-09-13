import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=16, base_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_ch, kernel_size=3, stride=2, padding=1)   # 8x8 -> 4x4
        self.conv2 = nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1)   # 4x4 -> 2x2
        self.flatten_dim = base_ch * 2 * 2 * 2
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_channels=1, latent_dim=16, base_ch=32):
        super().__init__()
        self.base_ch = base_ch
        self.fc = nn.Linear(latent_dim, base_ch * 2 * 2 * 2)
        self.deconv1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=3, stride=2,
                                           padding=1, output_padding=1)  # 2x2 -> 4x4
        self.deconv2 = nn.ConvTranspose2d(base_ch, out_channels, kernel_size=3, stride=2,
                                           padding=1, output_padding=1)  # 4x4 -> 8x8

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, self.base_ch * 2, 2, 2)
        h = F.relu(self.deconv1(h))
        x_recon = torch.sigmoid(self.deconv2(h))
        return x_recon


class VAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=16, base_ch=32):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, base_ch)
        self.decoder = Decoder(in_channels, latent_dim, base_ch)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def sample(self, n, device, latent_dim=16):
        z = torch.randn(n, latent_dim).to(device)
        with torch.no_grad():
            samples = self.decoder(z)
        return samples


def vae_loss(x_recon, x, mu, logvar, kl_weight=1.0):
    """ELBO loss = reconstruction loss + KL divergence."""
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum") / x.size(0)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_weight * kl_div, recon_loss, kl_div
