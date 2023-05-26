import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data import Dataset

class SST2Dataset(Dataset):
    def __init__(self, max_length):
        self.max_length = max_length
        self.dataset = load_dataset("sst2")
        self.embedding_matrix = self.load_embedding_matrix()

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
