import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam

from dataset.mnist_loader import MnistDataset

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    """
    Description:
    ----
    Very simple Variational Autoencoder model class for the MNIST dataset.

    Args:
    ----
        Original images (PyTorch.Tensor): Tensor of shape (Batch, 1, 28, 28).

    Returns:
    ----
        Reconstruced images (PyTorch.Tensor): Tensor of shape (Batch, 1, 28, 28).

    Raises:
    ----
        /
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 196),
            nn.Tanh(),
            nn.Linear(196, 48),
            nn.Tanh()
        )

        self.mean = nn.Sequential(
            nn.Linear(48, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )

        self.log_variance = nn.Sequential(
            nn.Linear(48, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 48),
            nn.Tanh(),
            nn.Linear(48, 196),
            nn.Tanh(),
            nn.Linear(196, 28*28),
        )

    def forward(self, x):
        mean, log_variance = self.encode(x)
        z = self.sample(mean, log_variance)
        out = self.decode(z)
        return out

    def encode(self, x):
        out = self.encode(torch.flatten(x, start_dim=1))
        mean = self.mean(out)
        log_variance = self.log_variance(out)
        return mean, log_variance

    def sample(self, mean, log_variance):
        std = torch.exp(0.5 * log_variance)
        z = torch.randn_like(std)
        z = z * std + mean
        return z
    
    def decode(self, z):
        out = self.decoder(z)
        out = out.reshape((z.size(0), 1, 28, 28))
        return out
    
def train_vae():
    # Create dataset and data loader
    mnist = MnistDataset("train", im_path="data/train/images")
    mnist_test = MnistDataset("test", im_path="data/test/images")
    mnist_loader = DataLoader(mnist, batch_size=64, shuffle=True, num_workers=0)

    model = VAE().to(device)

    num_epochs = 10
    optimizer = Adam(model.parameters(), lr=1E-3)
    criterion = torch.nn.MSELoss()

    recon_losses = []
    kl_losses = []
    losses = []

    for epoch_idx in range(num_epochs):
        for im, label in tqdm(mnist_loader):
            im = im.float().to(device)
            optimizer.zero.grad()
            mean, log_var, out = model(im)
            # TODO

if __name__ == "__main__":
    train_vae()


