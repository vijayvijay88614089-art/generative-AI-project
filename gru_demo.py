import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size=100, hidden_size=32, num_classes=2, bidirectional=False):
        super().__init__(); self.embed = nn.Embedding(vocab_size, 16); self.gru = nn.GRU(16, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.head = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)
    def forward(self, tokens):
        _, hidden = self.gru(self.embed(tokens))
        final = hidden[-2:].transpose(0, 1).flatten(1) if self.gru.bidirectional else hidden[-1]
        return self.head(final)
