import torch.nn as nn
import torch


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, label_size, hidden_dim):
        super().__init__()
        torch.manual_seed(42)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            input_size = embed_dim,
            hidden_size = hidden_dim,
            bidirectional = True,
            batch_first = True,
        )
        self.ffn = nn.Linear(in_features = hidden_dim * 2, out_features = label_size)
        self.softmax = nn.Softmax(dim = 2)

    def forward(self, input):
        rep = self.embedding(input)
        rep = self.bilstm(rep)[0]
        rep = self.ffn(rep)
        return self.softmax(rep)
