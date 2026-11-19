import torch.nn as nn


class DCGenerator(nn.Module):
    """Small 8x8 grayscale DCGAN generator, for architecture teaching only."""
    def __init__(self, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, 128 * 4 * 4), nn.ReLU(), nn.Unflatten(1, (128, 4, 4)),
                                 nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                                 nn.Conv2d(64, 1, 3, 1, 1), nn.Tanh())
    def forward(self, z): return self.net(z)


class DCDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 32, 3, 2, 1), nn.LeakyReLU(.2), nn.Conv2d(32, 64, 3, 2, 1),
                                 nn.LeakyReLU(.2), nn.Flatten(), nn.Linear(64 * 2 * 2, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x)
