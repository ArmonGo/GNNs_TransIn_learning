from Experiment import Experiment
from os import listdir
from os.path import isfile, join

exp_params = {
    'gnn_params' : 
    {
        'GCN' : {
                'hidden_channels': 16, 
                'out_channels': 1,
                'edge_dim' : 0 # no edge attributes 
                }, 
        'GAT' : {
                'hidden_channels': 16, 
                'out_channels': 1, 
                'heads': 3, 
                'edge_dim' : 0, # no edge attributes,
                # 'add_self_loop': False, default is true
                },
        'ARMA' : {
                'hidden_channels': 16, 
                'out_channels': 1, 
                'edge_dim' : 0 # no edge attributes
                },
        'TAG' : {
                'hidden_channels': 16, 
                'out_channels': 1, 
                'edge_dim' : 0 # no edge attributes
                },
        'GINE' : {
                'hidden_channels': 16, 
                'out_channels': 1, 
                'edge_dim' : 0 # no edge attributes
                },
        'SAGE' : {
                'hidden_channels': 16, 
                'out_channels': 1, 
                'edge_dim' : 0 # no edge attributes
                },
        'GIN' : {
                'hidden_channels': 16, 
                'out_channels': 1, 
                'edge_dim' : 0 # no edge attributes
                },
        'DEEPGCN' : {
                'hidden_channels': 16, 
                'out_channels': 1, 
                'edge_dim' : 0 # no edge attributes
                },
        'epochs' : 10000,
        'lr' : 0.001,
        'weight_decay' : 5e-4,
        'train_gnn_model_names' : ['GAT', 'GCN', 'ARMA', 'SAGE', 'GIN', 'DEEPGCN']
    },
   
    'data_path' : '',
    'data_label' : '',
    'model_save_path' : '.\\models\\',
    'performance_save_path' : '.\\rst\\'

}


graphs_data = [f for f in listdir('.\\data') if isfile(join('.\\data', f))]

for n in range(1,  len(graphs_data), 1):
    print(graphs_data[n], ' starts!')
    for k in ['transductive', 'inductive_weak','inductive_medium', 'inductive_strong']:
        exp_params['data_path'] = '.\\data\\' + graphs_data[n]
        exp_params['data_label'] = ('_').join(graphs_data[n].split('_')[:-1])
        exp_params['model_save_path'] = '.\\models\\' + k + '\\'
        exp_params['performance_save_path']  = '.\\rst\\' +  k + '\\'
        exp_params['graph_split_type'] = k
        exp = Experiment(exp_params)
        exp.run(repeat=30)

    