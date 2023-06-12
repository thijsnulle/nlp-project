import os
import time
from typing import List

import torch
import torch.nn as nn

from src.models.rnn_soft_attention import RNNWithSoftAttention
from src.sst2_dataset import SimpleTokenizer, SST2Dataset
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from scipy.spatial.distance import cosine

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class HotFlipAttack:
    def __init__(self, model: nn.Module, tokenizer: SimpleTokenizer, forbidden_tokens: List[str]):
        self.model = model
        self.tokenizer = tokenizer
        self.forbidden_tokens = forbidden_tokens

    def hotflip_attack(self, input_ids, true_label):
        original_mode = self.model.training
        self.model.train()

        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

        input_embeddings = self.model.get_input_embeddings()(input_ids)
        input_embeddings = input_embeddings.clone().detach().requires_grad_(True)

        outputs = self.model.forward_with_embeddings(input_embeddings)

        loss = nn.CrossEntropyLoss()(outputs, torch.tensor([true_label]).to(device))


        self.model.zero_grad()
        loss.backward()

        grads = input_embeddings.grad.data

        token_idx = torch.argmax(torch.sum(grads, dim=2))

        original_token = self.tokenizer.convert_id_to_word(input_ids[0, token_idx].cpu().numpy())

        best_replacement = self.get_best_replacement_token(grads[0, token_idx], original_token)

        input_ids = input_ids.cpu()
        input_ids[0, token_idx] = best_replacement
        self.model.train(original_mode)

        return input_ids

    def get_best_replacement_token(self, grads, original_token):
        # Add original token to the forbidden tokens for this iteration
        current_forbidden_tokens = self.forbidden_tokens.copy()
        current_forbidden_tokens.append(original_token)

        grads = grads.detach()

        original_token_id = self.tokenizer.word2idx[original_token]
        original_token_embedding = self.model.embedding(
            torch.tensor([original_token_id]).to(device)).detach()

        # Compute the cosine similarity between the original token embedding and all other token embeddings
        cosine_similarities = nn.functional.cosine_similarity(
            original_token_embedding,
            self.model.embedding.weight,
            dim=-1
        )

        # Compute the dot product between the gradients and all token embeddings
        dot_products = torch.matmul(grads, self.model.embedding.weight.t())

        # Create a boolean tensor that indicates whether each token is forbidden
        forbidden_tokens = torch.zeros_like(dot_products, dtype=torch.bool).to(device)
        for token, index in self.tokenizer.word2idx.items():
            if token in current_forbidden_tokens or token in stop_words:
                forbidden_tokens[index] = True

        # Find the token with the largest dot product that is not forbidden and has a cosine similarity above 0.8
        valid_tokens = (cosine_similarities >= 0.8) & ~forbidden_tokens
        valid_dot_products = dot_products[valid_tokens]
        if valid_dot_products.numel() == 0:  # no valid tokens
            # you can either return the original token id here,
            # or a random token id, depending on your preference.
            return self.tokenizer.word2idx[original_token]
        best_replacement_index = torch.argmax(valid_dot_products)
        best_replacement_token = torch.arange(dot_products.shape[0]).to(device)[valid_tokens][best_replacement_index]

        return best_replacement_token


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = SST2Dataset(max_length=128)
hidden_dim = 256
output_dim = 2
rnn_soft_attention = RNNWithSoftAttention(dataset.embedding_matrix, hidden_dim, output_dim)
rnn_soft_attention.load_state_dict(torch.load('models/RNNWithSoftAttention.pth'))
rnn_soft_attention = rnn_soft_attention.to(device)
attack = HotFlipAttack(model=rnn_soft_attention, tokenizer=dataset.tokenizer, forbidden_tokens=['great', 'awful', 'the', ''])

attack_dataset = dataset.dataset['test']

successes = 0
total = 0

for example in attack_dataset:
    input_text = example['sentence']
    true_label = example['label']
    input_ids = dataset.text_to_input_ids(input_text)
    input_ids = input_ids.to(device)
    rnn_input_ids = input_ids.unsqueeze(0).to(device)
    rnn_soft_attention.eval()
    with torch.no_grad():
        outputs = rnn_soft_attention(rnn_input_ids)
        predicted_label = outputs.argmax(dim=1).item()

    adversarial_input_ids = attack.hotflip_attack(input_ids, predicted_label)
    print(time.time())
    adversarial_text = ' '.join([dataset.tokenizer.convert_id_to_word(id) for id in adversarial_input_ids[0] if id != dataset.tokenizer.pad_token_id])

    rnn_soft_attention.eval()
    adversarial_input_ids = adversarial_input_ids.to(device)
    with torch.no_grad():
        adversarial_prediction = rnn_soft_attention(adversarial_input_ids).argmax(dim=1)

    if adversarial_prediction != predicted_label:
        successes += 1
    total += 1

print(f'Attack success rate: {successes / total * 100:.2f}%')
