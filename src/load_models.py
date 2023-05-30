import torch
from sst2_dataset import SST2Dataset
from models.rnn_soft_attention import RNNWithSoftAttention


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Load the RNNWithSoftAttention model from RNNWithSoftAttention.pth
    dataset = SST2Dataset(max_length=128)
    hidden_dim = 256
    output_dim = 2

    PATH = 'models/RNNWithSoftAttention.pth'
    model = RNNWithSoftAttention(dataset.embedding_matrix, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()

