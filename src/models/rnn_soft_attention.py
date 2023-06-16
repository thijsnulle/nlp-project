import torch
import torch.nn as nn
from textattack.models.helpers import LSTMForClassification


class RNNWithSoftAttention(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim):
        super(RNNWithSoftAttention, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        # Bi-directional RNN layer
        self.rnn = nn.GRU(embedding_matrix.size(1), hidden_dim, bidirectional=True, batch_first=True)

        # Attention layer
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Fully-connected layer for classification
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x=None, inputs_embeds=None):
        if x is not None:
            # Embedding layer
            embedded = self.embedding(x)
        elif inputs_embeds is not None:
            embedded = inputs_embeds
        else:
            raise ValueError("Either input tokens (x) or embeddings must be provided.")

        # RNN layer
        output, _ = self.rnn(embedded)

        # Attention layer
        attention_scores = self.attention(output).squeeze(-1)
        attention_weights = nn.functional.softmax(attention_scores, dim=1)
        weighted_sum = torch.sum(output * attention_weights.unsqueeze(-1), dim=1)

        # Fully-connected layer for classification
        logits = self.fc(weighted_sum)

        return logits

    def forward_with_embeddings(self, embeddings):
        # RNN layer
        output, _ = self.rnn(embeddings)

        # Attention layer
        attention_scores = self.attention(output).squeeze(-1)
        attention_weights = nn.functional.softmax(attention_scores, dim=1)
        weighted_sum = torch.sum(output * attention_weights.unsqueeze(-1), dim=1)

        # Fully-connected layer for classification
        logits = self.fc(weighted_sum)

        return logits

    def get_input_embeddings(self):
        return self.embedding

