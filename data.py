import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_digits


class DigitsDataset(Dataset):
    def __init__(self):
        digits = load_digits()
        images = digits.images.astype(np.float32) / 16.0  # scale to [0, 1]
        self.images = torch.from_numpy(images).unsqueeze(1)  # (N, 1, 8, 8)
        self.labels = digits.target

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_dataloader(batch_size=64, shuffle=True):
    dataset = DigitsDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def load_mnist_torchvision(root="./data", batch_size=128):
  
    import torchvision
    import torchvision.transforms as T

    transform = T.Compose([T.ToTensor()])
    train_set = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
    return DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
