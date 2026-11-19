import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, input_dim: int = 64, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.mu = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, input_dim), nn.Sigmoid())

    def forward(self, x):
        h = self.encoder(x); mu = self.mu(h); logvar = self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(.5 * logvar)
        return self.decoder(z), mu, logvar

    @staticmethod
    def loss(reconstruction, target, mu, logvar):
        recon = nn.functional.binary_cross_entropy(reconstruction, target, reduction="sum")
        kl = -.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + kl
