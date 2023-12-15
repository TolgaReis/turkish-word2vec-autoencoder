import torch
import torch.nn as nn


class Word2Vec2(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding1 = nn.Embedding(vocab_size, embedding_size)
        self.embedding2 = nn.Embedding(vocab_size, embedding_size)

    def forward(self, X_batch):
        U = self.embedding1(X_batch[:, 0])
        V = self.embedding2(X_batch[:, 1])

        output = torch.sigmoid(torch.sum(torch.mul(U, V), 1))
        return output
