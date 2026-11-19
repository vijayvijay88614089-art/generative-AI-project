from __future__ import annotations
import numpy as np

MODEL_CARDS = [
    {"name": "Hidden Markov Model (HMM)", "family": "Probabilistic sequence model", "definition": "A Markov chain over hidden states emits observed symbols with state-dependent probabilities.", "architecture": "Transition matrix A, emission matrix B, initial-state distribution π.", "working": "Forward/backward inference estimates state probabilities; Viterbi finds the most likely hidden-state path.", "uses": "Speech tagging, activity sequences, biological sequences.", "limits": "First-order Markov assumption and hand-designed discrete state structure."},
    {"name": "Autoencoder", "family": "Neural representation model", "definition": "An encoder compresses inputs into a latent vector; a decoder reconstructs the input.", "architecture": "Encoder → bottleneck z → decoder.", "working": "Minimize reconstruction loss L(x, x̂), often MSE or binary cross-entropy.", "uses": "Compression, denoising, anomaly detection.", "limits": "A plain autoencoder is not automatically a reliable generative sampler."},
    {"name": "Variational Autoencoder (VAE)", "family": "Latent-variable generative model", "definition": "Learns a probability distribution over latent representations and samples from it.", "architecture": "Encoder predicts μ and σ; reparameterized sample z=μ+σ⊙ε; decoder models x.", "working": "Optimize ELBO: reconstruction term minus KL divergence to a prior.", "uses": "Representation learning, controlled sampling, scientific data.", "limits": "May produce blurry samples; quality depends on model and objective."},
    {"name": "GAN", "family": "Adversarial generative model", "definition": "A generator creates samples while a discriminator learns to distinguish generated from real data.", "architecture": "G(z) ↔ D(x), trained with competing objectives.", "working": "G tries to raise D(G(z)); D tries to classify real and generated samples correctly.", "uses": "Image synthesis, augmentation, style transfer.", "limits": "Training instability, mode collapse, difficult evaluation."},
    {"name": "DCGAN", "family": "Convolutional GAN", "definition": "A GAN using convolutional layers for image representations.", "architecture": "Transposed convolutions in G; strided convolutions in D.", "working": "Convolutional inductive bias preserves spatial structure during adversarial learning.", "uses": "Small image generation demos.", "limits": "Still adversarially unstable and resolution-limited in simple setups."},
    {"name": "Conditional GAN (cGAN)", "family": "Conditional generation", "definition": "Conditions the generator and discriminator on a label or other side information.", "architecture": "G(z,y) and D(x,y).", "working": "The condition y steers which distribution the generator learns to sample.", "uses": "Class-conditioned images and controlled synthesis.", "limits": "Requires meaningful and balanced condition labels."},
    {"name": "StyleGAN", "family": "Style-based GAN", "definition": "A family of image generators that controls synthesis at multiple scales through style modulation.", "architecture": "Mapping network, style inputs, synthesis network, per-layer modulation.", "working": "Different latent/style components influence coarse-to-fine visual attributes.", "uses": "High-resolution face and artistic image synthesis.", "limits": "Compute-intensive; realistic outputs can enable deceptive deepfakes."},
    {"name": "CycleGAN", "family": "Unpaired image translation", "definition": "Learns mappings between two image domains without paired examples.", "architecture": "Two generators and two discriminators with adversarial and cycle-consistency losses.", "working": "Translate A→B→A and penalize changes that fail to reconstruct the original.", "uses": "Seasonal or artistic domain translation, research prototypes.", "limits": "Cycle consistency does not guarantee semantic correctness or safe use."},
]


def sample_markov(seed: int = 7, length: int = 24):
    """Tiny transparent two-state HMM-style sequence sample, not a trained model."""
    rng = np.random.default_rng(seed)
    states = [0]
    observations = []
    transition = np.array([[.85, .15], [.25, .75]])
    emission = np.array([[.8, .2], [.2, .8]])
    for _ in range(length):
        state = states[-1]
        observations.append(int(rng.choice([0, 1], p=emission[state])))
        states.append(int(rng.choice([0, 1], p=transition[state])))
    return states[:-1], observations


def attention_matrix(tokens: list[str]):
    # Toy scaled dot-product attention over deterministic token vectors.
    token_codes = [sum((j + 1) * ord(ch) for j, ch in enumerate(str(token))) for token in tokens]
    vectors = np.array([[((code * (i + 3)) % 29) / 29 for i in range(8)] for code in token_codes], dtype=float)
    scores = vectors @ vectors.T / np.sqrt(vectors.shape[1])
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores); weights /= weights.sum(axis=1, keepdims=True)
    return weights
