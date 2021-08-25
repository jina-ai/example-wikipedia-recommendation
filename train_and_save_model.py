import model
from model import GCN

import torch
from torch_geometric.data import InMemoryDataset, Data, download_url
from torch_geometric.datasets import wikics
import pandas as pd
import json


def train(model, data):
      model.train()
      optimizer.zero_grad() 
      out = model(data.x, data.edge_index) 
      loss = criterion(out, data.y)  # Compute the loss solely based on the training nodes.  
      loss.backward()  
      optimizer.step()  
      return loss

def test(model, data):
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc


if __name__ == '__main__':
    n_epochs = 400

    dataset = wikics.WikiCS('./wiki-cs-dataset_autodownload')
    metadata = json.load(open('wiki-cs-dataset/dataset/metadata.json'))

    num_classes = len(dataset.data.y.unique())
    num_features = dataset.data.x.shape[1]

    model = GCN(num_node_features=num_features, 
                num_classes=num_classes, 
                hidden_channels=128)

    data = dataset.data 
    loss_values = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs):
        loss = train(model, data)
        loss_values.append(loss)
        print(f'\rEpoch: {epoch:03d}, Loss: {loss:.4f}', end='')

    test_acc = test(model, data)
    print(f'Test Accuracy: {test_acc:.4f}')
    torch.save(model.state_dict(), './model/saved_model.torch')

  