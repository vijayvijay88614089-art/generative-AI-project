import torch
from torch import nn


class ConditionalGenerator(nn.Module):
    def __init__(self, classes=3, z_dim=16, output_dim=64):
        super().__init__()
        self.embed = nn.Embedding(classes, 8)
        self.net = nn.Sequential(nn.Linear(z_dim + 8, 64), nn.ReLU(), nn.Linear(64, output_dim), nn.Tanh())
    def forward(self, noise, labels): return self.net(torch.cat([noise, self.embed(labels)], dim=1))


class ConditionalDiscriminator(nn.Module):
    def __init__(self, classes=3, input_dim=64):
        super().__init__()
        self.embed = nn.Embedding(classes, 8)
        self.net = nn.Sequential(nn.Linear(input_dim + 8, 64), nn.LeakyReLU(.2), nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, samples, labels): return self.net(torch.cat([samples, self.embed(labels)], dim=1))
