from typing import Any, Dict

from OpenAttack import Attacker, Classifier, Victim
import OpenAttack.attackers
import torch
import torch.nn as nn
import OpenAttack as oa
import os
import ssl

from OpenAttack.attackers import HotFlipAttacker

ssl._create_default_https_context = ssl._create_unverified_context

from src.models.embedding_model import EmbeddingModel
from src.models.rnn_soft_attention import RNNWithSoftAttention
from src.models.self_attention import SelfAttentionModel
from src.sst2_dataset import SST2Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomClassifier(Classifier):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def get_pred(self, input_):
        input_ids = self.dataset.text_to_input_ids(input_[0]).unsqueeze(0).to(device)
        outputs = self.model(input_ids)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()

    def get_prob(self, input_):
        input_ids = self.dataset.text_to_input_ids(input_[0]).unsqueeze(0).to(device)
        outputs = nn.functional.softmax(self.model(input_ids), dim=1)
        return outputs.cpu().detach().numpy()[0]


class CustomHotFlipAttacker(HotFlipAttacker):

    def __call__(self, victim: Victim, input_: Dict[str, Any]):
        change_info = super().__call__(victim, input_)
        print(f"Original word: {input_['x']}")
        print(f"Changed word: {change_info}")
        return change_info


# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = SST2Dataset(max_length=128)
hidden_dim = 256
output_dim = 2
rnn_soft_attention = RNNWithSoftAttention(dataset.embedding_matrix, hidden_dim, output_dim).to(device)
rnn_soft_attention = torch.load('../models/RNNWithSoftAttention.pth')
num_samples = 10
sst2_data = []
for i in range(num_samples):
    example = dataset.dataset['train'][i]
    sst2_data.append({'x': example['sentence'], 'y': example['label']})

custom_classifier = CustomClassifier(rnn_soft_attention, dataset)
attacker = CustomHotFlipAttacker()
attack_eval = OpenAttack.AttackEval(attacker, custom_classifier)
attack_eval.eval(sst2_data, visualize=True)
