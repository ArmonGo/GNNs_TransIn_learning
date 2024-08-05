
from sklearn.preprocessing import QuantileTransformer
import torch 
from torch_geometric.utils import subgraph, add_self_loops, remove_self_loops
import copy 
from torch_geometric.data import Data

def scale_graph(train_G, val_G, test_G, scaler = QuantileTransformer):
    # x 
    if (train_G.x.max(dim=0)[0]!=train_G.x.min(dim=0)[0]).sum() ==0:
        pass
    else:
        sl = scaler()
        train_G.x = torch.tensor(sl.fit_transform(train_G.x.numpy()))
        val_G.x = torch.tensor(sl.transform(val_G.x.numpy()))
        test_G.x = torch.tensor(sl.transform(test_G.x.numpy()))
    # edge attr
    if hasattr(train_G, 'edge_attr') and  train_G.edge_attr is not None:
        if (train_G.edge_attr.max(dim=0)[0]!=train_G.edge_attr.min(dim=0)[0]).sum()==0:
            pass
        else:
            sl = scaler()
            train_G.edge_attr = torch.tensor(sl.fit_transform(train_G.edge_attr.numpy()))
            val_G.edge_attr = torch.tensor(sl.transform(val_G.edge_attr.numpy()))
            test_G.edge_attr = torch.tensor(sl.transform(test_G.edge_attr.numpy()))
    return train_G, val_G, test_G


def pad_self_loop(G):
    nodes = torch.tensor(list(range(len(G.y))))
    filter = ~torch.isin(nodes, G.edge_index[0])
    if filter.sum()>0:
        pad_edges = torch.concat((nodes[filter], nodes[filter])).reshape(2, -1)
        print('pad_edges', pad_edges)
        G.edge_index = torch.concat((G.edge_index,pad_edges ), dim = 1)
        if G.edge_attr is not None:
            pad_edge_attr = torch.ones(pad_edges.shape[0], G.edge_attr.shape[1])
            G.edge_attr = torch.concat((G.edge_attr, pad_edge_attr ), dim = 0)

def sample_nodes(nodes, sample_num):
    if sample_num == 'all' or sample_num == 0 :
        return nodes
    else:
        assert sample_num > 0 
        indices = torch.randperm(len(nodes))
        return nodes[indices[: sample_num]]

def inductive_graph_construct(sample_num, G, train_v, val_v, test_v):
    sub_edges, _ = subgraph(train_v, G.edge_index, relabel_nodes = True)
    train_G = Data(x = G.x[train_v], edge_index=sub_edges, y= G.y[train_v],  train_mask = (torch.ones(len(train_v)) == 1))
    if sample_num == 'all': # weak inductive, no need to drop nodes
        train_val_v, _ = torch.concat((train_v, val_v)).sort()
        sub_edges, _ = subgraph(train_val_v, G.edge_index, relabel_nodes = True)
        val_G = Data(x = G.x[train_val_v], edge_index=sub_edges, y= G.y[train_val_v], val_mask = torch.isin(train_val_v, val_v))
        test_G = copy.deepcopy(G)
        test_G.test_mask =  torch.isin(torch.tensor(range(test_G.x.shape[0])), test_v)
        
    if sample_num == 0: # strong inductive, only use the sample nodes and sample edges
        sub_edges, _ = subgraph(val_v, G.edge_index, relabel_nodes = True)
        val_G = Data(x = G.x[val_v], edge_index=sub_edges, y= G.y[val_v], val_mask = (torch.ones(len(val_v)) == 1))
        sub_edges, _ = subgraph(test_v, G.edge_index, relabel_nodes = True)
        test_G = Data(x = G.x[test_v], edge_index=sub_edges, y= G.y[test_v], test_mask = (torch.ones(len(test_v)) == 1))
    
    else: # medium inductive, sample nodes for percentage
        sample_train_v = sample_nodes(train_v, sample_num)
        train_val_v, _ = torch.concat((sample_train_v, val_v)).sort()
        sub_edges, _ = subgraph(train_val_v, G.edge_index, relabel_nodes = True)
        val_G = Data(x = G.x[train_val_v], edge_index=sub_edges, y= G.y[train_val_v], val_mask = torch.isin(train_val_v, val_v))

        sample_train_val_v = sample_nodes(torch.concat((train_v, val_v)), sample_num)
        train_val_test_v, _ = torch.concat((sample_train_val_v, test_v)).sort()
        sub_edges, _ = subgraph(train_val_test_v, G.edge_index, relabel_nodes = True)
        test_G = Data(x = G.x[train_val_test_v], edge_index=sub_edges, y= G.y[train_val_test_v], test_mask = torch.isin(train_val_test_v, test_v))
    return (train_G, val_G, test_G)


def train_val_test_split(G, training_size = 0.6, validation_size = 0.2, add_self_loop = False):
    graph_set = {'transductive' : None, 
                 'inductive_weak' : None, 
                 'inductive_medium' : None, 
                 'inductive_strong' : None
                 }
    if G.x is None:
        G.x = torch.ones((len(G.y), 1)) # assign node features as 1
    if add_self_loop:
        G.edge_index = remove_self_loops(G.edge_index)[0] # remove and then add at the end 
        if G.edge_attr is None or not hasattr(G, 'edge_attr'):
            G.edge_index = add_self_loops(G.edge_index)[0]
        else:
            G.edge_index, G.edge_attr = add_self_loops(G.edge_index,G.edge_attr, fill_value='mean')
        pad_self_loop(G)
    # split trainin, validation, test nodes
    indices = torch.randperm(len(G.x))
    indices_train  = indices[: int(len(G.x)* training_size)]
    indices_val = indices[int(len(G.x) * training_size) : int(len(G.x) * (training_size + validation_size))]
    indices_test = indices[int(len(G.x) * (training_size + validation_size)):]
    train_v, ix = indices_train.sort()
    val_v, ix = indices_val.sort()
    test_v, ix = indices_test.sort()
    # first transductive learning - masking 
    G_trans = copy.deepcopy(G)
    G_trans.train_mask = torch.isin(torch.tensor(range(G_trans.x.shape[0])), train_v)
    G_trans.val_mask = torch.isin(torch.tensor(range(G_trans.x.shape[0])), val_v)
    G_trans.test_mask = torch.isin(torch.tensor(range(G_trans.x.shape[0])), test_v)
    graph_set['transductive'] = (G_trans, G_trans, G_trans) # train/val/test are the same
    # inductive nodes
    graph_set['inductive_weak'] =  inductive_graph_construct('all', G, train_v, val_v, test_v)
    graph_set['inductive_medium'] =  inductive_graph_construct(len(train_v) - len(val_v), G, train_v, val_v, test_v)
    graph_set['inductive_strong'] =  inductive_graph_construct(0, G, train_v, val_v, test_v)
    return graph_set


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, save_best_path = None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.save_best_path = save_best_path

    def early_stop(self, validation_loss, best_model = None):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            if best_model is not None:
                assert self.save_best_path is not None and self.save_best_path != ''
                torch.save(best_model, self.save_best_path)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False