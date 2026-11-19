import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size=100, hidden_size=32, num_classes=2, bidirectional=False):
        super().__init__(); self.embed = nn.Embedding(vocab_size, 16); self.lstm = nn.LSTM(16, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.head = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)
    def forward(self, tokens):
        _, (hidden, _) = self.lstm(self.embed(tokens))
        final = hidden[-2:].transpose(0, 1).flatten(1) if self.lstm.bidirectional else hidden[-1]
        return self.head(final)
