import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data import Dataset


class SimpleTokenizer:
    def __init__(self, word2idx):
        self.word2idx = word2idx
        self.pad_token_id = len(word2idx) - 1  # assuming the last index is reserved for padding

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        input_ids = [self.pad_token_id] * len(tokens)
        for i, word in enumerate(tokens):
            if word in self.word2idx:
                input_ids[i] = self.word2idx[word]
        return input_ids

    def convert_id_to_word(self, id):
        return list(self.word2idx.keys())[list(self.word2idx.values()).index(id)]


class SST2Dataset(Dataset):
    def __init__(self, max_length):
        self.max_length = max_length
        self.dataset = load_dataset("sst2")
        self.embedding_matrix = self.load_embedding_matrix()
        self.tokenizer = SimpleTokenizer(self.word2idx)

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

    def text_to_input_ids(self, text):
        input_ids = [self.embedding_matrix.size(0) - 1] * self.max_length
        for i, word in enumerate(text.split()[:self.max_length]):
            if word in self.word2idx:
                input_ids[i] = self.word2idx[word]
        return torch.tensor(input_ids)

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
