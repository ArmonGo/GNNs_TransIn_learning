import torch 
from Models import GAT, GCN,  ARMA, DEEPGCN, SAGE, GIN
import torch.nn as nn
from utils import EarlyStopper


class Experiment:
    def __init__(self, exp_params):
        self.gnn_params = exp_params['gnn_params']
        self.data_path = exp_params['data_path']
        self.data_label = exp_params['data_label']
        self.model_save_path = exp_params['model_save_path']
        self.performance_save_path = exp_params['performance_save_path']
        self.graph_split_type = exp_params['graph_split_type']
        self.G_train, self.G_val, self.G_test = torch.load(self.data_path)[self.graph_split_type]
        
        # multi labels or multi class or binary 
        self.multi_label = False
        if len(self.G_train.y.shape)>1:
            self.classes = self.G_train.y.shape[1]
            if self.G_train.y.sum(dim = 1).max()>1:
                self.multi_label = True
        else:
            self.classes = len(set(torch.tensor(self.G_train.y).numpy()))
        # reshape the y_true
        if self.classes == 2 and not self.multi_label:
            self.y_true_train = self.G_train.y[self.G_train.train_mask].reshape(-1, 1).float().to('cuda')
            self.y_true_val = self.G_val.y[self.G_val.val_mask].reshape(-1, 1).float().to('cuda')
        else:
            self.y_true_train = self.G_train.y[self.G_train.train_mask].to('cuda')
            self.y_true_val = self.G_val.y[self.G_val.val_mask].to('cuda')
        # fill in the param
        for i in self.gnn_params:
            if i in ['GCN', 'GAT', 'ARMA', 'SAGE', 'GIN', 'DEEPGCN']:
                self.gnn_params[i]['in_channels'] = self.G_train.x.shape[1]
                if self.classes ==2:
                    self.gnn_params[i]['out_channels'] = 1
                else:
                    self.gnn_params[i]['out_channels'] = self.classes
        self.G_train.to('cuda')
        self.G_val.to('cuda')
        self.G_test.to('cuda')

    def fit_gnn(self, gnn_model_name):
        if gnn_model_name == 'GAT':
            model = GAT(**self.gnn_params['GAT']).to('cuda')
        if gnn_model_name == 'GCN':
            model = GCN(**self.gnn_params['GCN']).to('cuda')
        if gnn_model_name == 'ARMA':
            model = ARMA(**self.gnn_params['ARMA']).to('cuda')
        if gnn_model_name == 'SAGE':
            model = SAGE(**self.gnn_params['SAGE']).to('cuda')
        if gnn_model_name == 'DEEPGCN5':
            model = DEEPGCN(**self.gnn_params['DEEPGCN']).to('cuda')
        if gnn_model_name == 'GIN':
            model = GIN(**self.gnn_params['GIN']).to('cuda')
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.gnn_params['lr'], weight_decay=self.gnn_params['weight_decay'])
        early_stopper = EarlyStopper(patience=100, min_delta=0, save_best_path= self.model_save_path + '\\' + self.data_label + '_' + gnn_model_name + '.pt')
        # loss function
        if self.classes == 2 or self.multi_label:
            loss_f = nn.BCELoss()
        elif  self.classes >2:
            loss_f = nn.CrossEntropyLoss()
        # output funct
        if self.multi_label or self.classes ==2:
            self.out_f = nn.Sigmoid()
        else:
            self.out_f = nn.Softmax()

        for i in range(self.gnn_params['epochs']):
            out = self.out_f(model(self.G_train.x, self.G_train.edge_index)[self.G_train.train_mask])
            loss = loss_f(out, self.y_true_train)
            with torch.no_grad():
                out_val = self.out_f(model(self.G_val.x, self.G_val.edge_index)[self.G_val.val_mask])
                loss_val = loss_f(out_val, self.y_true_val)
                if early_stopper.early_stop(loss_val.detach(), model):
                    print('Early stop epochs: ', i, ' min validation loss: ', early_stopper.min_validation_loss)             
                    break
            if i % 200==0:
                print("training loss: ", loss.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict_gnn(self, gnn_model_name):
        model = torch.load(self.model_save_path + '\\' + self.data_label + '_'  + gnn_model_name + '.pt')
        with torch.no_grad():
            prob_y = self.out_f(model(self.G_test.x, self.G_test.edge_index))
        return prob_y.to('cpu')
    
    def run(self, repeat=1):
        performance = {}
        # run gnn benchmark first
        for n in range(repeat):
            print('repeat: ', n)
            performance[n] = {}
            for m in self.gnn_params['train_gnn_model_names']:
                print(m, ' starts fitting')
                self.fit_gnn(gnn_model_name = m)
                performance[n][m] = (self.predict_gnn(gnn_model_name = m), self.G_test.test_mask.to('cpu')) # add the mask 
            torch.save(performance, self.performance_save_path + self.data_label + '_'  + 'test_performance.pt')
        return 'done!'
        
