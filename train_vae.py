import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from vae import VAE, vae_loss
from data import get_dataloader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 8
LATENT_DIM = 16
EPOCHS = 60
BATCH_SIZE = 64
LR = 1e-3

os.makedirs("outputs", exist_ok=True)


def train():
    loader = get_dataloader(batch_size=BATCH_SIZE)
    model = VAE(in_channels=1, latent_dim=LATENT_DIM, base_ch=32).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = []
    for epoch in range(1, EPOCHS + 1):
        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
        for x, _ in loader:
            x = x.to(DEVICE)
            x_recon, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x_recon, x, mu, logvar, kl_weight=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()

        n_batches = len(loader)
        avg_loss = total_loss / n_batches
        history.append(avg_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"[VAE] Epoch {epoch:3d}/{EPOCHS} | "
                  f"ELBO loss: {avg_loss:.4f} | recon: {total_recon/n_batches:.4f} | "
                  f"KL: {total_kl/n_batches:.4f}")

    torch.save(model.state_dict(), "outputs/vae.pt")

    # Save loss curve
    plt.figure()
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("ELBO loss")
    plt.title("VAE training loss")
    plt.savefig("outputs/vae_loss_curve.png")
    plt.close()

    # Save sample generations
    model.eval()
    samples = model.sample(64, DEVICE, LATENT_DIM).cpu()
    fig, axes = plt.subplots(8, 8, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i, 0], cmap="gray")
        ax.axis("off")
    plt.suptitle("VAE generated samples")
    plt.savefig("outputs/vae_samples.png")
    plt.close()

    print("VAE training complete. Model + samples saved to outputs/")
    return history


if __name__ == "__main__":
    train()
