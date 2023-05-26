import torch
import torch.nn as nn

from models.embedding_model import EmbeddingModel
from models.rnn_soft_attention import RNNWithSoftAttention
from models.self_attention import SelfAttentionModel
from sst2_dataset import SST2Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def training_loop(model: nn.Module, dataset: SST2Dataset, data_split_ratio=0.8, num_epochs=10, lr=0.001, batch_size=128):
    train_size = int(len(dataset) * data_split_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(num_epochs):
        model.train()

        print(f'Epoch {epoch + 1} of {num_epochs}')

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0

            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Epoch {epoch + 1}: Accuracy = {100 * correct / total:.2f}%')
    
    print(f'Finished Training for {model.__class__.__name__}')

    torch.save(model.state_dict(), f'models/{model.__class__.__name__}.pth')

if __name__ == '__main__':
    dataset = SST2Dataset(max_length=128)
    hidden_dim = 256
    output_dim = 2

    embedding_model = EmbeddingModel(dataset.embedding_matrix, hidden_dim, output_dim).to(device)
    rnn_soft_attention = RNNWithSoftAttention(dataset.embedding_matrix, hidden_dim, output_dim).to(device)
    self_attention = SelfAttentionModel(output_dim, dataset.embedding_matrix).to(device)

    training_loop(rnn_soft_attention, dataset, num_epochs=5)
    training_loop(embedding_model, dataset, num_epochs=100)
    
    # training_loop(self_attention, dataset, num_epochs=5)
