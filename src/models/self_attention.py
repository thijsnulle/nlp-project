import math

import torch
import torch.nn as nn


class SelfAttentionModel(nn.Module):
    def __init__(self, num_classes, embedding_matrix, max_length):
        super(SelfAttentionModel, self).__init__()

        self.embedding_dim = embedding_matrix.size(1)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.positional_encoding = PositionalEncoding(self.embedding_dim, max_length)
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=5)
            for _ in range(5)  # Number of self-attention layers
        ])
        self.fc = nn.Linear(self.embedding_dim, num_classes)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x=None, inputs_embeds=None):
        if x is not None:
            # Apply the embeddings but don't apply positional encoding yet
            embedded_input = self.embedding(x)
        elif inputs_embeds is not None:
            embedded_input = inputs_embeds
        else:
            raise ValueError("Either input tokens (x) or embeddings must be provided.")

        # Apply positional encoding to a detached copy
        positional_encoded_input = self.positional_encoding(embedded_input.clone().detach())

        for self_attention_layer in self.self_attention_layers:
            attention_output, _ = self_attention_layer(positional_encoded_input, positional_encoded_input,
                                                       positional_encoded_input)
            positional_encoded_input = attention_output + positional_encoded_input  # Residual connection

        summed_output = torch.sum(positional_encoded_input, dim=1)
        logits = self.fc(summed_output)

        return logits

    def forward_with_embeddings(self, embeddings):
        # Apply positional encoding to a detached copy
        positional_encoded_input = self.positional_encoding(embeddings)

        for self_attention_layer in self.self_attention_layers:
            attention_output, _ = self_attention_layer(positional_encoded_input, positional_encoded_input,
                                                       positional_encoded_input)
            positional_encoded_input = attention_output + positional_encoded_input  # Residual connection

        summed_output = torch.sum(positional_encoded_input, dim=1)
        logits = self.fc(summed_output)

        return logits

    def get_input_embeddings(self):
        return self.embedding

    def get_gradients(self, input_ids, labels):
        self.zero_grad()  # Clear any previously stored gradients
        logits, _ = self.forward(input_ids)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()  # Compute gradients

        gradients = []
        for param in self.parameters():
            if param.grad is not None:
                gradients.append(param.grad.clone())
            else:
                gradients.append(None)

        return gradients

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=128):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

