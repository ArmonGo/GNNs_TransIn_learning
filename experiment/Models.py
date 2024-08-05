import torch 
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ARMAConv, GENConv, GINConv
from torch import Tensor


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim = 0):
        super().__init__()
        if edge_dim>0:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(edge_dim, hidden_channels), torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, 1), torch.nn.ReLU()
            )
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)


    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor = None) -> Tensor:
        if edge_attr is not None:
            edge_weight = self.mlp(edge_attr)
            x = self.conv1(x, edge_index, edge_weight).relu()
            x = self.conv2(x, edge_index, edge_weight).relu()
        else:
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
        x = self.linear(x)
        return x
    
class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim = 0):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)


    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor = None) -> Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.linear(x)
        return x

    
class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim = 0):
        super().__init__()
        self.mlp_conv1 =  torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels), torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU()
        )
        self.mlp_conv2 =  torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(),
           
        )
        self.conv1 = GINConv(self.mlp_conv1, train_eps = True)
        self.conv2 = GINConv(self.mlp_conv2, train_eps = False) 
        self.linear = torch.nn.Linear(hidden_channels, out_channels)


    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor = None) -> Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.linear(x)
        return x   
    

class ARMA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim = 0):
        super().__init__()
        if edge_dim >0:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(edge_dim, hidden_channels), torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, 1), torch.nn.ReLU()
            )
        self.conv1 = ARMAConv(in_channels, hidden_channels, num_layers=2) # 2 layers directly
        self.linear = torch.nn.Linear(hidden_channels, out_channels)


    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor = None) -> Tensor:
        if edge_attr is not None:
            edge_weight = self.mlp(edge_attr)
            x = self.conv1(x, edge_index, edge_weight).relu()
        else:
            x = self.conv1(x, edge_index).relu()
        x = self.linear(x)
        return x
   
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, edge_dim = 0):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, edge_dim =edge_dim)
        self.conv2 = GATConv(hidden_channels*heads, hidden_channels, heads=1, edge_dim =edge_dim) # output channel 1
        self.linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor = None) -> Tensor:
        if edge_attr is not None:
            x = self.conv1(x, edge_index, edge_attr).relu()
            x = self.conv2(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
        x = self.linear(x)
        return x
    
    
class DEEPGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim =0):
        super().__init__()
        self.conv1 = GENConv(in_channels, hidden_channels, num_layers=2, edge_dim =edge_dim)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor = None) -> Tensor:
        if edge_attr is not None:
            x = self.conv1(x, edge_index, edge_attr).relu()
        else:
            x = self.conv1(x, edge_index).relu()
        x = self.linear(x)
        return x
    