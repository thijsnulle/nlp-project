import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np
import torch.nn.functional as F
import math

# COLAB link (it requires a lot of GPU memory): https://colab.research.google.com/drive/1aWOcPPMRrMLK5afDhipP4eYsG6m3ktpM#scrollTo=HVrzka4Tc1wB

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

    def forward(self, input_ids):
        embedded_input = self.embedding(input_ids)
        positional_encoded_input = self.positional_encoding(embedded_input)

        for self_attention_layer in self.self_attention_layers:
            attention_output, _ = self_attention_layer(positional_encoded_input, positional_encoded_input, positional_encoded_input)
            positional_encoded_input = attention_output + positional_encoded_input  # Residual connection

        summed_output = torch.sum(positional_encoded_input, dim=1)
        logits = self.fc(summed_output)
        return logits


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


class SST2Dataset(Dataset):
    def __init__(self, max_length):
        self.max_length = max_length
        self.dataset = self.load_dataset()
        self.embedding_matrix = self.load_embedding_matrix()

    def load_dataset(self):
        dataset = load_dataset("sst2")
        return dataset

    def load_embedding_matrix(self):
        word2idx = {}
        embedding_matrix = []

        with open('glove.6B.100d.txt', 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')

                word2idx[word] = len(word2idx)
                embedding_matrix.append(vector)

        self.word2idx = word2idx

        return torch.tensor(embedding_matrix, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset['train'])

    def __getitem__(self, idx):
        example = self.dataset['train'][idx]
        input_text = example['sentence']
        label = example['label']

        input_ids = [self.embedding_matrix.size(0) - 1] * self.max_length
        for i, word in enumerate(input_text.split()[:self.max_length]):
            if word in self.word2idx:
                input_ids[i] = self.word2idx[word]

        return torch.tensor(input_ids), label


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 2
max_length = 256
batch_size_local = 512
num_epochs = 50
learning_rate = 0.001

train_size_proportion = 0.8

dataset = SST2Dataset(max_length)
model = SelfAttentionModel(num_classes, dataset.embedding_matrix, max_length).to(device)


train_size = int(train_size_proportion * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


train_loader = DataLoader(train_dataset, batch_size=batch_size_local, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size_local, shuffle=False, drop_last=True)


for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation on the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Update total and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {accuracy:.2f}%")