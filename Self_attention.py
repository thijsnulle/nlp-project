import torch
import torch.nn as nn

class SelfAttentionModel(nn.Module):
    def __init__(self, num_classes, embedding_matrix):
        super(SelfAttentionModel, self).__init__()

        self.embedding_dim = embedding_matrix.size(1)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.self_attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=5)
        self.fc = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, input_ids):
        embedded_input = self.embedding(input_ids)
        attention_output, _ = self.self_attention(embedded_input, embedded_input, embedded_input)
        attention_output = attention_output.permute(1, 0, 2)
        summed_output = torch.sum(attention_output, dim=1)
        logits = self.fc(summed_output)
        return logits