import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim: int = 16, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, 64), nn.LeakyReLU(.2), nn.Linear(64, 128), nn.LeakyReLU(.2), nn.Linear(128, output_dim), nn.Tanh())
    def forward(self, z): return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.LeakyReLU(.2), nn.Linear(128, 64), nn.LeakyReLU(.2), nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x)
