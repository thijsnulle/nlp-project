import textattack
import torch
import torch.nn as nn

from src.models.rnn_soft_attention import RNNWithSoftAttention


class RNNWithSoftAttentionTextAttack(textattack.models.helpers.LSTMForClassification):
    def __init__(self, embedding_matrix, hidden_dim, output_dim):
        super().__init__()
        self.rnn_soft_attention = RNNWithSoftAttention(embedding_matrix, hidden_dim, output_dim)

    def forward(self, x=None, inputs_embeds=None):
        return self.rnn_soft_attention(x, inputs_embeds)

    def forward_with_embeddings(self, embeddings):
        return self.rnn_soft_attention.forward_with_embeddings(embeddings)

    def get_input_embeddings(self):
        return self.rnn_soft_attention.get_input_embeddings()

    def get_grad(self, loss):
        return self.rnn_soft_attention.get_grad(loss)

    def get_word_embeddings(self):
        return self.rnn_soft_attention.get_word_embeddings()
