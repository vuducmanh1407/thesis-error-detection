import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool # For graph-level tasks

class GCNWithWeights(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super(GCNWithWeights, self).__init__()
        self.num_layers = num_layers

        self.conv_in = Linear(in_channels, hidden_channels)
        self.gcn_list = list()
        for i in range(self.num_layers):
            self.gcn_list.append(GCNConv(hidden_channels, hidden_channels))
        self.gcn_list = Sequential(*self.gcn_list)
        self.conv_out = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # Pass edge_weight to the conv layers
        x = self.conv_in(x)
        for i in range(self.num_layers):
            x = self.gcn_list[i](x, edge_index, edge_weight).relu()
            x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv_out(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.conv2(x, edge_index, edge_weight)

        x = global_mean_pool(x, batch)

        return x