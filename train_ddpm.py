import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from ddpm import SimpleUNet, DiffusionSchedule, ddpm_loss, sample_ddpm
from data import get_dataloader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 8
TIMESTEPS = 200
EPOCHS = 60
BATCH_SIZE = 64
LR = 2e-3

os.makedirs("outputs", exist_ok=True)


def train():
    loader = get_dataloader(batch_size=BATCH_SIZE)
    model = SimpleUNet(in_channels=1, base_ch=32, time_dim=64).to(DEVICE)
    schedule = DiffusionSchedule(timesteps=TIMESTEPS, device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = []
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for x, _ in loader:
            x = x.to(DEVICE)
            loss = ddpm_loss(model, schedule, x, DEVICE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        history.append(avg_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"[DDPM] Epoch {epoch:3d}/{EPOCHS} | noise MSE loss: {avg_loss:.5f}")

    torch.save(model.state_dict(), "outputs/ddpm.pt")

    # Save loss curve
    plt.figure()
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("Noise-prediction MSE loss")
    plt.title("DDPM training loss")
    plt.savefig("outputs/ddpm_loss_curve.png")
    plt.close()

    # Save sample generations (this loops over all diffusion timesteps, so it's slower)
    print("Sampling from DDPM (reverse diffusion over", TIMESTEPS, "steps)...")
    samples = sample_ddpm(model, schedule, n_samples=64, img_size=IMG_SIZE, device=DEVICE).cpu()
    fig, axes = plt.subplots(8, 8, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i, 0], cmap="gray")
        ax.axis("off")
    plt.suptitle("DDPM generated samples")
    plt.savefig("outputs/ddpm_samples.png")
    plt.close()

    print("DDPM training complete. Model + samples saved to outputs/")
    return history


if __name__ == "__main__":
    train()
