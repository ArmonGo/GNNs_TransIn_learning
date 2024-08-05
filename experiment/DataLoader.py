from utils import * 
import torch 

def get_pyg_data(data_name, extractor_name, data_extractor, training_size, validation_size,  add_self_loop =False):
    if data_name is not None:
        dataset = data_extractor(root='/tmp/' + data_name, name=data_name)
        G = dataset.data
    else:
        dataset  = data_extractor(root='/tmp/' + extractor_name)
        G = dataset.data
    graph_set = train_val_test_split(G, training_size = training_size, 
                                                    validation_size = validation_size, 
                                                    add_self_loop = add_self_loop)
    f_n = extractor_name + '_' + str(data_name)  + '_graph.pt'
    for k, v in graph_set.items():
            graph_set[k] = scale_graph(v[0], v[1], v[2])
    torch.save(graph_set, '.\\data\\' + f_n) 
    return 'done'


def get_pyg_data_description(data_name, extractor_name, data_extractor):
    if data_name is not None:
        dataset = data_extractor(root='/tmp/' + data_name, name=data_name)
        G = dataset.data
    else:
        dataset  = data_extractor(root='/tmp/' + extractor_name)
        G = dataset.data
    print(str(extractor_name) + '-' +str(data_name), G)
    return G