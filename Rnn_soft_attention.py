import torch
import torch.nn as nn

class RNNWithSoftAttention(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim):
        super(RNNWithSoftAttention, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)

        # Bi-directional RNN layer
        self.rnn = nn.GRU(embedding_matrix.size(1), hidden_dim, bidirectional=True, batch_first=True)

        # Attention layer
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Fully-connected layer for classification
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)

        # RNN layer
        output, _ = self.rnn(embedded)

        # Attention layer
        attention_scores = self.attention(output).squeeze(-1)
        attention_weights = nn.functional.softmax(attention_scores, dim=1)
        weighted_sum = torch.sum(output * attention_weights.unsqueeze(-1), dim=1)

        # Fully-connected layer for classification
        logits = self.fc(weighted_sum)

        return logits
