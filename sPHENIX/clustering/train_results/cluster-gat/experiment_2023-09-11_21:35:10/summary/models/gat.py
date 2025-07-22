import torch_geometric.nn as gnn
import torch.nn as nn
import torch

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = gnn.Sequential('x, edge_index, edge_weights', [
            (gnn.GATv2Conv(input_dim, hidden_dim, fill_value=0, edge_dim=1), 'x, edge_index, edge_weights -> x'),
            (nn.ReLU(), 'x -> x'),
            (gnn.GATv2Conv(hidden_dim, hidden_dim, fill_value=0, edge_dim=1), 'x, edge_index, edge_weights -> x'),
            (nn.ReLU(), 'x -> x'),
            (gnn.GATv2Conv(hidden_dim, hidden_dim, fill_value=0, edge_dim=1), 'x, edge_index, edge_weights -> x'),
            (nn.Linear(hidden_dim, 1), 'x -> x')
        ])


    def forward(self, data):
        x = data.x
        edge_weights = 1/data.edge_attr
        edge_weights[data.edge_attr == 0] = 1/1e-4
        edge_index = data.edge_index

        return self.network(x, edge_index, edge_weights).squeeze(-1)
