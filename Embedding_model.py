import torch
import torch.nn as nn

class EmbeddingModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim):
        super(EmbeddingModel, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.fc = nn.Linear(embedding_matrix.size(1), hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        x = self.output(x)

        return x
