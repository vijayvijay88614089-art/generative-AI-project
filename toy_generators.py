"""Small CPU teaching demos using the bundled 8x8 handwritten-digits dataset."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import load_digits
from torch import nn

from .autoencoder import Autoencoder
from .cgan import ConditionalDiscriminator, ConditionalGenerator
from .dcgan import DCDiscriminator, DCGenerator
from .gan import Discriminator, Generator
from .vae import VAE


def handwritten_digits(n: int = 1200):
    """Return a deterministic subset of sklearn's bundled 8x8 digit images.

    Pixels are normalized from [0, 16] to [0, 1] for the autoencoder/VAE and
    labels span all ten digit classes for the conditional GAN.
    """
    dataset = load_digits()
    rng = np.random.default_rng(22)
    indices = rng.permutation(len(dataset.data))[:max(1, min(int(n), len(dataset.data)))]
    images = dataset.data[indices].astype(np.float32) / 16.0
    labels = dataset.target[indices].astype(np.int64)
    return torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def _validate_steps(steps: int) -> int:
    steps = int(steps)
    if not 1 <= steps <= 1000:
        raise ValueError("Training steps must be between 1 and 1,000.")
    return steps


def train_demo(name: str, steps: int = 80):
    """Train one small model and return sample images plus plotted loss history."""
    steps = _validate_steps(steps)
    torch.manual_seed(4)
    torch.set_num_threads(1)
    x, y = handwritten_digits()

    if name == "Autoencoder":
        model = Autoencoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        history = []
        for _ in range(steps):
            reconstruction, _ = model(x)
            loss = nn.functional.binary_cross_entropy(reconstruction, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
        with torch.no_grad():
            output, _ = model(x[:6])
        return {
            "input": x[:6].numpy().reshape(-1, 8, 8),
            "output": output.numpy().reshape(-1, 8, 8),
            "loss": history[-1],
            "loss_history": {"Reconstruction": history},
        }

    if name == "VAE":
        model = VAE()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        history = []
        for _ in range(steps):
            reconstruction, mu, logvar = model(x)
            loss = VAE.loss(reconstruction, x, mu, logvar) / len(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
        with torch.no_grad():
            samples = model.decoder(torch.randn(6, 8)).numpy().reshape(-1, 8, 8)
        return {"output": samples, "loss": history[-1], "loss_history": {"VAE objective": history}}

    if name in {"GAN", "DCGAN", "Conditional GAN"}:
        bce = nn.BCELoss()
        generator_losses, discriminator_losses = [], []
        if name == "GAN":
            real = x * 2 - 1
            generator, discriminator = Generator(), Discriminator()
            latent_dim = 16
            generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.99))
            discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.99))
            for _ in range(steps):
                fake = generator(torch.randn(32, latent_dim)).detach()
                real_batch = real[torch.randint(len(real), (32,))]
                d_loss = bce(discriminator(real_batch), torch.ones(32, 1)) + bce(discriminator(fake), torch.zeros(32, 1))
                discriminator_optimizer.zero_grad()
                d_loss.backward()
                discriminator_optimizer.step()
                g_loss = bce(discriminator(generator(torch.randn(32, latent_dim))), torch.ones(32, 1))
                generator_optimizer.zero_grad()
                g_loss.backward()
                generator_optimizer.step()
                discriminator_losses.append(float(d_loss.detach()))
                generator_losses.append(float(g_loss.detach()))
            with torch.no_grad():
                output = ((generator(torch.randn(6, latent_dim)) + 1) / 2).clamp(0, 1).numpy().reshape(-1, 8, 8)

        elif name == "DCGAN":
            real = x.reshape(-1, 1, 8, 8) * 2 - 1
            generator, discriminator = DCGenerator(), DCDiscriminator()
            latent_dim = 64
            generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.99))
            discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.99))
            for _ in range(steps):
                real_batch = real[torch.randint(len(real), (32,))]
                fake = generator(torch.randn(32, latent_dim)).detach()
                d_loss = bce(discriminator(real_batch), torch.ones(32, 1)) + bce(discriminator(fake), torch.zeros(32, 1))
                discriminator_optimizer.zero_grad()
                d_loss.backward()
                discriminator_optimizer.step()
                g_loss = bce(discriminator(generator(torch.randn(32, latent_dim))), torch.ones(32, 1))
                generator_optimizer.zero_grad()
                g_loss.backward()
                generator_optimizer.step()
                discriminator_losses.append(float(d_loss.detach()))
                generator_losses.append(float(g_loss.detach()))
            with torch.no_grad():
                output = ((generator(torch.randn(6, latent_dim)) + 1) / 2).clamp(0, 1).numpy().reshape(-1, 8, 8)

        else:
            real = x * 2 - 1
            generator = ConditionalGenerator(classes=10)
            discriminator = ConditionalDiscriminator(classes=10)
            generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.99))
            discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.99))
            for _ in range(steps):
                indices = torch.randint(len(x), (32,))
                batch_labels = y[indices]
                fake = generator(torch.randn(32, 16), batch_labels).detach()
                d_loss = bce(discriminator(real[indices], batch_labels), torch.ones(32, 1)) + bce(
                    discriminator(fake, batch_labels), torch.zeros(32, 1)
                )
                discriminator_optimizer.zero_grad()
                d_loss.backward()
                discriminator_optimizer.step()
                labels = torch.arange(6).remainder(10)
                g_loss = bce(discriminator(generator(torch.randn(6, 16), labels), labels), torch.ones(6, 1))
                generator_optimizer.zero_grad()
                g_loss.backward()
                generator_optimizer.step()
                discriminator_losses.append(float(d_loss.detach()))
                generator_losses.append(float(g_loss.detach()))
            labels = torch.arange(6).remainder(10)
            with torch.no_grad():
                output = ((generator(torch.randn(6, 16), labels) + 1) / 2).clamp(0, 1).numpy().reshape(-1, 8, 8)
            return {
                "output": output,
                "labels": labels.numpy(),
                "loss": generator_losses[-1],
                "loss_history": {"Generator": generator_losses, "Discriminator": discriminator_losses},
            }

        return {
            "output": output,
            "loss": generator_losses[-1],
            "loss_history": {"Generator": generator_losses, "Discriminator": discriminator_losses},
        }

    if name == "CycleGAN concept":
        images = x[:6].numpy().reshape(-1, 8, 8)
        # A reversible transpose illustrates cycle consistency; this is not training.
        translated = np.transpose(images, (0, 2, 1))
        cycle = np.transpose(translated, (0, 2, 1))
        error = float(np.abs(images - cycle).mean())
        return {"input": images, "output": translated, "cycle": cycle, "loss": error}

    raise ValueError(f"Unknown generative model demo: {name}")
