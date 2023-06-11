from abc import ABC

import torch
import torch.nn as nn
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import ModelWrapper
from textattack.transformations import WordSwapEmbedding, WordSwapGradientBased
from textattack.search_methods import GreedySearch, BeamSearch
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.goal_functions import UntargetedClassification
from textattack.attack_recipes import AttackRecipe
from textattack.attack_recipes import HotFlipEbrahimi2017
from torchtext.datasets import sst2
from torchtext.vocab import GloVe

from transformers import AutoTokenizer

from src.models.RNNLSTM import RNNWithSoftAttentionTextAttack
from src.models.rnn_soft_attention import RNNWithSoftAttention
from src.sst2_dataset import SST2Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomTextAttackModelWrapper(ModelWrapper):
    def __init__(self, model, dataset, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.gradients = None
        self.dataset = dataset

    def __call__(self, text_inputs):
        outputs = []
        for text_input in text_inputs:
            input_ids = self.dataset.text_to_input_ids(text_input).unsqueeze(0).to(device)
            output = nn.functional.softmax(self.model(input_ids), dim=1)
            outputs.append(output.detach().cpu().numpy()[0])
        return outputs

    def save_gradients(self, grad_input):
        self.gradients = grad_input[0]

    def get_grad(self, input_text):
        # Convert the input text to tokens
        tokens = self.dataset.text_to_input_ids(input_text)

        # Convert the tokens to their corresponding IDs using the tokenizer
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Convert the input IDs to a tensor and move it to the appropriate device
        input_ids = torch.tensor([input_ids]).to(device)

        # Get the embeddings from the model
        # Call the get_input_embeddings method only once and store the result
        input_embeddings = self.model.get_input_embeddings()
        embeddings = input_embeddings(input_ids).clone()
        embeddings = embeddings.clone().detach().requires_grad_(True)

        # Register the backward hook here
        embeddings.register_hook(self.save_gradients)
        # Clone and require grad here

        # Run the model on the embeddings
        output = self.model(inputs_embeds=embeddings)

        # Create dummy variable and compute loss
        target = torch.zeros(output.size(0), dtype=torch.long).to(device)
        loss = nn.functional.cross_entropy(output, target)

        # Zero out the gradients
        self.model.zero_grad()

        # Backward pass
        loss.backward()

        return {'gradient': self.gradients.detach().cpu().numpy(), 'ids': input_ids[0].cpu().numpy()}


class CustomHotFlipAttack(AttackRecipe):
    def __init__(self, model):
        transformation = WordSwapGradientBased(model, top_n=1)
        search_method = BeamSearch()
        constraints = [RepeatModification(), StopwordModification(), MaxWordsPerturbed(max_num_words=1),
                       WordEmbeddingDistance(min_cos_sim=0.8), PartOfSpeech()]
        goal_function = UntargetedClassification(model)

        super().__init__(goal_function, constraints, transformation, search_method)

    def build(self, model):
        return self


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = SST2Dataset(max_length=128)
hidden_dim = 256
output_dim = 2
rnn_soft_attention = RNNWithSoftAttentionTextAttack(dataset.embedding_matrix, hidden_dim, output_dim).to(device)
rnn_soft_attention.load_state_dict(torch.load('models/RNNWithSoftAttentionTextAttack.pth'))

model_wrapper = CustomTextAttackModelWrapper(rnn_soft_attention, dataset, dataset.tokenizer)
rnn_soft_attention.get_input_embeddings().register_backward_hook(model_wrapper.save_gradients)

# Initialize the attack
attack = CustomHotFlipAttack(model_wrapper)

# Prepare the dataset
Huggindataset = HuggingFaceDataset("sst2", None, "test")


def predict_label(model, tokenizer, text):
    model.eval()

    with torch.no_grad():
        input_ids = tokenizer.text_to_input_ids(text).unsqueeze(0).to(device)
        output = model(input_ids)
        _, predicted_label = torch.max(output, dim=1)

    return predicted_label.item()


results = []
for example in Huggindataset:
    input_text, ground_truth = example
    ground_truth = predict_label(rnn_soft_attention, dataset, input_text['sentence'])
    rnn_soft_attention.train()
    print(input_text, ground_truth)
    results.append(attack.attack(input_text['sentence'], ground_truth))
    print(results)

print(results)
