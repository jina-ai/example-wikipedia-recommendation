import torch
from torch_geometric.nn import GCNConv
from torch.functional import F


class GCN(torch.nn.Module):
    def __init__(self, num_node_features=300, num_classes=10, hidden_channels=128):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def encode(self, x, edge_index):
        feature_map = None

        def get_activation(model, model_inputs, output):
            nonlocal feature_map
            feature_map = output.detach()

        handle = self.conv1.register_forward_hook(get_activation)
        self.forward(x, edge_index)
        handle.remove()
        return feature_map

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
