import torch
from torch import nn


class Autoencoder(nn.Module):
    """Small fully connected autoencoder for flattened, [0, 1] inputs.

    The default width of 64 matches the bundled 8x8 handwritten-digit images.
    ``forward`` returns ``(reconstruction, latent)`` so the demo can display
    both the reconstructed input and its compressed representation.
    """

    def __init__(self, input_dim: int = 64, latent_dim: int = 8):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be a positive integer.")
        if latent_dim < 1 or latent_dim >= input_dim:
            raise ValueError("latent_dim must be positive and smaller than input_dim.")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, input_dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(f"Expected a 2D batch shaped (batch, {self.input_dim}); got {tuple(x.shape)}.")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features per row; got {x.shape[1]}.")
        if not x.is_floating_point():
            raise TypeError("Autoencoder inputs must use a floating-point dtype.")
        z = self.encoder(x)
        return self.decoder(z), z
