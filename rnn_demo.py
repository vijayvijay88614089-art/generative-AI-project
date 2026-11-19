import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size=100, hidden_size=32, num_classes=2):
        super().__init__(); self.embed = nn.Embedding(vocab_size, 16); self.rnn = nn.RNN(16, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, num_classes)
    def forward(self, tokens):
        _, hidden = self.rnn(self.embed(tokens)); return self.head(hidden[-1])
