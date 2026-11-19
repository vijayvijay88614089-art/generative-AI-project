import torch
from torch import nn


class TinyTransformer(nn.Module):
    """Small causal character transformer; intended for toy data, not useful language generation."""
    def __init__(self, vocab_size: int, width: int = 32, heads: int = 4, max_len: int = 128):
        super().__init__()
        self.token = nn.Embedding(vocab_size, width); self.position = nn.Embedding(max_len, width)
        layer = nn.TransformerEncoderLayer(width, heads, width * 2, batch_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, num_layers=2); self.head = nn.Linear(width, vocab_size)
    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device)
        h = self.token(x) + self.position(pos)
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device), diagonal=1).bool()
        return self.head(self.blocks(h, mask=mask))
