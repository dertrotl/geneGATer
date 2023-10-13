import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv as GATConv


class GAT_linear_negbin(torch.nn.Module):
    """GAT model architecture with linear output layer and negative binomial loss."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=1, add_self_loops=False)

        self.conv2 = GATConv(
            hidden_channels, out_channels, heads=1, concat=False, return_attention_weights=True, add_self_loops=False
        )

        self.conv_var = GATConv(hidden_channels, out_channels)

        self.linear_mean = torch.nn.Linear(out_channels, out_channels, bias=True)
        # self.linear_var = torch.nn.Linear(out_channels, out_channels, bias = True)

    def forward(self, x, edge_index):
        """Forward pass of the model.

        Dropout layer, elu, dropout, conv2 layer, relu, linear layer, conv for var estimation.
        """
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        mean, alpha = self.conv2(x, edge_index, return_attention_weights=True)
        mean = F.relu(mean)
        var = self.conv_var(x, edge_index)

        mean = self.linear_mean(mean)
        # var = self.linear_var(var)
        return [mean, var], alpha


class GAT_linear(torch.nn.Module):
    """GAT model architecture with linear output layer and MSE/Poisson loss."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=1, add_self_loops=False)

        self.conv2 = GATConv(
            hidden_channels, out_channels, heads=1, concat=False, return_attention_weights=True, add_self_loops=False
        )

        self.linear = torch.nn.Linear(out_channels, out_channels, bias=True)

    def forward(self, x, edge_index):
        """Forward pass of the model.

        Dropout layer, elu, dropout, conv2 layer, relu, linear layer.
        """
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x, alpha = self.conv2(x, edge_index, return_attention_weights=True)
        x = F.relu(x)

        # mean = F.relu( self.linear(x))
        mean = self.linear(x)

        return mean, alpha


class GAT_negbin(torch.nn.Module):
    """GAT model architecture with negative binomial loss."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=1, add_self_loops=False)

        self.conv2 = GATConv(
            hidden_channels, out_channels, heads=1, concat=False, return_attention_weights=True, add_self_loops=False
        )

        # self.conv_var = GATConv(hidden_channels, out_channels)
        self.conv_var = GATConv(out_channels, out_channels)
        # self.conv_var = torch.nn.Linear(out_channels, out_channels, bias = True)

    def forward(self, x, edge_index):
        """Forward pass of the model.

        Dropout layer, relu, dropout, conv2 layer, conv2 layer for var estimation.
        """
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        mean, alpha = self.conv2(x, edge_index, return_attention_weights=True)
        # mean = F.relu(mean)
        # var = F.relu(self.conv_var(x, edge_index))
        # var = self.conv_var(x, edge_index)
        var = self.conv_var(mean, edge_index)
        # var = self.conv_var(mean)
        return [mean, var], alpha


class GAT(torch.nn.Module):
    """GAT model architecture with MSE/Poisson loss."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=1, add_self_loops=False)

        self.conv2 = GATConv(
            hidden_channels, out_channels, heads=1, concat=False, return_attention_weights=True, add_self_loops=False
        )

    def forward(self, x, edge_index):
        """Forward pass of the model.

        Dropout layer, elu, dropout, conv2 layer.
        """
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x, alpha = self.conv2(x, edge_index, return_attention_weights=True)
        # x = F.relu(x)

        return x, alpha
