import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class DiffusionSchedule:
    def __init__(self, timesteps=200, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x0, t, noise):
        """Forward process: sample x_t given x0 and timestep t (closed form)."""
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1m_ac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ac * x0 + sqrt_1m_ac * noise



class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(1, out_ch)
        self.norm2 = nn.GroupNorm(1, out_ch)

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h


class SimpleUNet(nn.Module):

    def __init__(self, in_channels=1, base_ch=32, time_dim=64):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)

        self.down1 = ConvBlock(in_channels, base_ch, time_dim)
        self.down2 = ConvBlock(base_ch, base_ch * 2, time_dim)
        self.pool = nn.AvgPool2d(2)

        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 2, time_dim)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_conv1 = ConvBlock(base_ch * 2 + base_ch, base_ch, time_dim)
        self.out_conv = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

        h1 = self.down1(x, t_emb)              # full res
        h2 = self.down2(self.pool(h1), t_emb)   # half res
        b = self.bottleneck(h2, t_emb)

        u = self.up1(b)
        # handle odd sizes (e.g., 8x8 -> pooled 4x4 -> upsampled 8x8, matches)
        if u.shape[-1] != h1.shape[-1]:
            u = F.interpolate(u, size=h1.shape[-2:], mode="nearest")
        u = torch.cat([u, h1], dim=1)
        u = self.up_conv1(u, t_emb)
        return self.out_conv(u)



@torch.no_grad()
def sample_ddpm(model, schedule: DiffusionSchedule, n_samples, img_size, device):
    model.eval()
    x = torch.randn(n_samples, 1, img_size, img_size).to(device)
    for t_step in reversed(range(schedule.timesteps)):
        t = torch.full((n_samples,), t_step, device=device, dtype=torch.long)
        eps_pred = model(x, t)

        alpha = schedule.alphas[t_step]
        alpha_cumprod = schedule.alphas_cumprod[t_step]
        beta = schedule.betas[t_step]

        mean = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_cumprod)) * eps_pred
        )

        if t_step > 0:
            noise = torch.randn_like(x)
            var = schedule.posterior_variance[t_step]
            x = mean + torch.sqrt(var) * noise
        else:
            x = mean
    model.train()
    return x.clamp(0, 1)


def ddpm_loss(model, schedule: DiffusionSchedule, x0, device):
    """Standard DDPM training objective: predict the noise added at a random t."""
    batch_size = x0.size(0)
    t = torch.randint(0, schedule.timesteps, (batch_size,), device=device).long()
    noise = torch.randn_like(x0)
    x_t = schedule.q_sample(x0, t, noise)
    noise_pred = model(x_t, t)
    return F.mse_loss(noise_pred, noise)
