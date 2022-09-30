import numpy as np
import dgl
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import roc_auc_score
import wandb
import warnings
warnings.filterwarnings("ignore")
# from einops import rearrange, repeat

# ## Dataloader

from dgl.data import DGLDataset

class MDEVSPGraphDataset(DGLDataset):
    def __init__(self, list_graphs, label='sol_int',dossier='', rang=None, cc=True):
        self.dossier = dossier
        self.list_graphs = list_graphs
        self.label = label
        self.rang=rang
        self.cc = cc
        super().__init__(name='MDEVSPgraph')
        

    def process(self):
        self.graphs = []

        for pb in self.list_graphs:
            
            df_nodes = pd.read_csv(self.dossier+'DGLgraphs_column/nodes'+pb+'.csv',sep='\t',index_col = 0)
            df_edges = pd.read_csv(self.dossier+'DGLgraphs_column/edges'+pb+'.csv',sep='\t',index_col = 0)
            
            t_min = df_nodes.loc[df_nodes.t_s>0,'t_s'].min()
            df_nodes.loc[df_nodes.o==1,'t_s'] = t_min
            df_nodes.loc[df_nodes.o==1,'t_e'] = t_min
            df_nodes['ts_n'] = (df_nodes['t_s']-t_min)/(df_nodes['t_s'].max()-t_min)
            df_nodes['te_n'] = (df_nodes['t_e']-t_min)/(df_nodes['t_e'].max()-t_min)
            df_nodes['duree'] = df_nodes['t_e']-df_nodes['t_s']
            df_nodes['duree_n'] = df_nodes['duree']/df_nodes['duree'].max()
            df_nodes['nb_trip'] = len(df_nodes.loc[df_nodes.n==1])

            df_edges['c_n'] = df_edges['c']/df_edges['c'].max()
            df_edges['r1_n'] = df_edges['r1']/df_edges['r1'].max()
            df_edges['td_n'] = df_edges['t_d']/df_edges['t_d'].max()
            df_edges['tw_n'] = df_edges['t_w']/df_edges['t_w'].max()
            df_edges['delta_n'] = df_edges['delta']/df_edges['delta'].max()
            df_edges['type_edge'] = df_edges['src'].apply(lambda x:x[0])+'_'+df_edges['dst'].apply(lambda x:x[0])
            df_edges['mask'] = True
            df_edges.loc[df_edges['type_edge'].isin(['o_n','n_k','w_w','w_c','c_c','c_w']),'mask'] = False

            if self.rang is None :
                if not self.cc :
                    df_edges = df_edges.loc[df_edges.type_edge.isin(['n_n','n_w','w_n','o_n','n_k','w_k','w_w'])].reset_index(drop=True)
            else :
                df_edges['rang_c'] = df_edges.groupby("src")["c"].rank("min", ascending=True)
                if self.cc :
                    df_edges = df_edges.loc[(df_edges.rang_c<=self.rang)].reset_index(drop=True)
                else :
                    df_edges = df_edges.loc[(df_edges.rang_c<=self.rang)&(df_edges.type_edge.isin(['n_n','n_w','w_n','o_n','n_k','w_k','w_w']))].reset_index(drop=True)
            g = dgl.graph((df_edges['idx_src'], df_edges['idx_dst']), num_nodes=len(df_nodes))
            g.edata['features'] = torch.tensor(df_edges[['c_n', 'r1_n', 'td_n', 'tw_n', 'delta_n', 'rg_id','rg']].astype('float').values) #'c', 'r1', 't_d', 't_w', 'delta', 'c_n', 'r1_n', 'td_n', 'tw_n', 'delta_n', 'rg_id','rg'
            g.edata['mask'] = torch.tensor(df_edges['mask'].values)
            g.edata['label'] = torch.tensor(df_edges[self.label].values)
            g.ndata['features'] = torch.tensor(df_nodes[[x for x in df_nodes.columns if x in ['o', 'k', 'n', 'w', 'c', 'nb_dep_10', 'nb_dep_id_10','nb_fin_10', 'nb_fin_id_10', 'ts_n', 'te_n', 'duree_n'] or 's_' in x or 'e_' in x]].astype('float').values) #if x in ['t_s', 't_e','o', 'k', 'n', 'w', 'c', 'd', 'nb_dep_10', 'nb_dep_id_10','nb_fin_10', 'nb_fin_id_10', 'ts_n', 'te_n', 'duree_n'] or 's_' in x or 'e_' in x 
            self.graphs.append(g)

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)

# ## Modèle
class Layer(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.U = nn.Linear(d, d,bias=False)
        self.V = nn.Linear(d, d,bias=False)
        self.A = nn.Linear(d, d,bias=False)
        self.B = nn.Linear(d, d,bias=False)
        self.C = nn.Linear(d, d,bias=False)
        self.bn_h = nn.BatchNorm1d(d, affine=False)
        self.bn_e = nn.BatchNorm1d(d, affine=False)
    
    def forward(self, g, h, e):
        with g.local_scope():
            g.ndata['h'] = self.V(h)
            g.edata['w'] = torch.sigmoid(e)
            # update_all is a message passing API.
            g.update_all(message_func=dgl.function.u_mul_e('h', 'w', 'm'), 
                reduce_func=dgl.function.max('m', 'agg'))
            agg = g.ndata['agg']

            out_h = h + torch.relu(self.bn_h(self.U(h)+agg))
            out_e = torch.matmul(torch.transpose(g.inc('in'),0,1),self.B(h)
                                )+torch.matmul(torch.transpose(g.inc('out'),0,1),self.C(h))
            out_e = e +  torch.relu(self.bn_e(self.A(e)+out_e))
            return out_h, out_e

class Model(nn.Module):
    def __init__(self, f_n, f_e, d, L, dropout=False):
        super().__init__()
        self.emb_n = nn.Linear(f_n, d)
        self.emb_e = nn.Linear(f_e, d)
        for l in range(L):
            self.add_module(f'layer_{l}', Layer(d))
        self.W1 = nn.Linear(3*d, d)
        self.W2 = nn.Linear(d, 1)
        self.act = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.01)
        self.drop = dropout

    def forward(self, g, h, e, return_logit = False):
        h = self.emb_n(h)
        e = self.emb_e(e)
        for n, layer in self.named_children():
            if 'layer' in n:
                if self.drop :
                    h, e = self.dropout(h), self.dropout(e)
                h, e = layer(g, h, e)
        moy = 1/h.shape[-2]*torch.sum(h,dim=0).repeat(e.shape[-2],1)
        hi = torch.matmul(torch.transpose(g.inc('out'),0,1),h)
        hj = torch.matmul(torch.transpose(g.inc('in'),0,1),h)
        out = torch.concat([moy, hi,hj],dim=1)
        out = self.W1(out)
        out = self.relu(out)
        out = self.W2(out)
        out = out[:,0]
        if return_logit :
            return out
        return self.act(out)

class ModelK(nn.Module):
    def __init__(self, f_n, f_e, d, L, K, dropout=False):
        super().__init__()
        self.emb_n = nn.Linear(f_n, d)
        self.emb_e = nn.Linear(f_e, d)
        for l in range(L):
            self.add_module(f'layer_{l}', Layer(d))
        self.W0 = nn.Linear(3*d, d)
        for k in range(K):
            self.add_module(f'W_{k}', nn.Linear(d,d))
        self.Wf = nn.Linear(d, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.01)
        self.drop = dropout
        self.act = nn.Sigmoid()

    def forward(self, g, h, e, return_logit = False):
        h = self.emb_n(h)
        e = self.emb_e(e)
        for n, layer in self.named_children():
            if 'layer' in n:
                if self.drop :
                    h, e = self.dropout(h), self.dropout(e)
                h, e = layer(g, h, e)
        moy = 1/h.shape[-2]*torch.sum(h,dim=0).repeat(e.shape[-2],1)
        hi = torch.matmul(torch.transpose(g.inc('out'),0,1),h)
        hj = torch.matmul(torch.transpose(g.inc('in'),0,1),h)
        out = torch.concat([moy, hi,hj],dim=1)
        out = self.W0(out)
        out = self.relu(out)
        for n, layer in self.named_children():
            if 'W_' in n:
                out = layer(out)
                out = self.relu(out)
        out = self.Wf(out)
        out = out[:,0]
        return self.act(out)

class ModelSoftmaxSimple(nn.Module):
    def __init__(self, f_n, f_e, d, L, dropout=False):
        super().__init__()
        self.emb_n = nn.Linear(f_n, d)
        self.emb_e = nn.Linear(f_e, d)
        for l in range(L):
            self.add_module(f'layer_{l}', Layer(d))
        self.W0 = nn.Linear(3*d, d)
        self.Wf = nn.Linear(d, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.01)
        self.drop = dropout

    def forward(self, g, h, e, return_logit = False):
        h = self.emb_n(h)
        e = self.emb_e(e)
        for n, layer in self.named_children():
            if 'layer' in n:
                if self.drop :
                    h, e = self.dropout(h), self.dropout(e)
                h, e = layer(g, h, e)
        moy = 1/h.shape[-2]*torch.sum(h,dim=0).repeat(e.shape[-2],1)
        hi = torch.matmul(torch.transpose(g.inc('out'),0,1),h)
        hj = torch.matmul(torch.transpose(g.inc('in'),0,1),h)
        out = torch.concat([moy, hi,hj],dim=1)
        out = self.W0(out)
        out = self.relu(out)
        out = self.Wf(out)
        out = out[:,0]
        M = g.inc('out')
        O = torch.sparse_coo_tensor(M._indices(), out, M.size())
        O = torch.sparse.softmax(O, dim=1)
        Q = torch.zeros_like(out)
        Q[O.coalesce().indices()[1]] = O.coalesce().values()
        if return_logit :
            return out
        return Q

class ModelSoftmax(nn.Module):
    def __init__(self, f_n, f_e, d, L, K, dropout=False):
        super().__init__()
        self.emb_n = nn.Linear(f_n, d)
        self.emb_e = nn.Linear(f_e, d)
        for l in range(L):
            self.add_module(f'layer_{l}', Layer(d))
        self.W0 = nn.Linear(3*d, d)
        for k in range(K):
            self.add_module(f'W_{k}', nn.Linear(d,d))
        self.Wf = nn.Linear(d, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.01)
        self.drop = dropout

    def forward(self, g, h, e, return_logit = False):
        h = self.emb_n(h)
        e = self.emb_e(e)
        for n, layer in self.named_children():
            if 'layer' in n:
                if self.drop :
                    h, e = self.dropout(h), self.dropout(e)
                h, e = layer(g, h, e)
        moy = 1/h.shape[-2]*torch.sum(h,dim=0).repeat(e.shape[-2],1)
        hi = torch.matmul(torch.transpose(g.inc('out'),0,1),h)
        hj = torch.matmul(torch.transpose(g.inc('in'),0,1),h)
        out = torch.concat([moy, hi,hj],dim=1)
        out = self.W0(out)
        out = self.relu(out)
        for n, layer in self.named_children():
            if 'W_' in n:
                out = layer(out)
                out = self.relu(out)
        out = self.Wf(out)
        out = out[:,0]
        M = g.inc('out')
        O = torch.sparse_coo_tensor(M._indices(), out, M.size())
        O = torch.sparse.softmax(O, dim=1)
        Q = torch.zeros_like(out)
        Q[O.coalesce().indices()[1]] = O.coalesce().values()
        if return_logit :
            return out
        return Q

class ModelAttention(nn.Module):
    def __init__(self, f_n, f_e, d, L, M):
        super().__init__()
        self.M = M
        self.d_k = d//M
        self.emb_n = nn.Linear(f_n, d)
        self.emb_e = nn.Linear(f_e, d)
        for l in range(L):
            self.add_module(f'layer_{l}', Layer(d))
        self.GAT = dgl.nn.GATv2Conv(d, self.d_k,self.M,allow_zero_in_degree=True)
        self.O = nn.Linear(d,d)
        self.W1 = nn.Linear(3*d, d)
        self.W2 = nn.Linear(d, 1)
        self.act = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, g, h, e):
        h = self.emb_n(h)
        e = self.emb_e(e)
        for n, layer in self.named_children():
            if 'layer' in n:
                h, e = layer(g, h, e)
        out = self.GAT(g, h)
        out = rearrange(out, 'n M v-> n (M v)')
        h = self.O(out)

        nb_batch = g.batch_size
        batch_nodes = g.batch_num_nodes().tolist()
        batch_edges = g.batch_num_edges().tolist()
        h_graph = torch.split(h, batch_nodes)
        moy = [torch.mean(x, dim=0) for x in h_graph] # devrait être (b, d_h)
        moy = torch.cat([moy[i].repeat(batch_edges[i],1) for i in range(nb_batch)],dim=0)

        hi = torch.matmul(torch.transpose(g.inc('out'),0,1),h)
        hj = torch.matmul(torch.transpose(g.inc('in'),0,1),h)
        out = torch.concat([moy, hi,hj],dim=1)
        out = self.W1(out)
        out = self.relu(out)
        out = self.W2(out)
        return self.act(out)[:,0]
        
class ModelAttentionSimple(nn.Module):
    def __init__(self, f_n, f_e, d, L, M):
        super().__init__()
        self.M = M
        self.d_k = d//M
        self.emb_n = nn.Linear(f_n, d)
        self.emb_e = nn.Linear(f_e, d)
        for l in range(L):
            self.add_module(f'layer_{l}', Layer(d))
        self.GAT = dgl.nn.GATv2Conv(d, self.d_k,self.M,allow_zero_in_degree=True)
        self.O = nn.Linear(d,d)
        self.W1 = nn.Linear(3*d, d)
        self.W2 = nn.Linear(d, 1)
        self.act = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, g, h, e):
        h = self.emb_n(h)
        e = self.emb_e(e)
        for n, layer in self.named_children():
            if 'layer' in n:
                h, e = layer(g, h, e)
        out = self.GAT(g, h)
        out = rearrange(out, 'n M v-> n (M v)')
        h = self.O(out)

        moy = 1/h.shape[-2]*torch.sum(h,dim=0).repeat(e.shape[-2],1)
        hi = torch.matmul(torch.transpose(g.inc('out'),0,1),h)
        hj = torch.matmul(torch.transpose(g.inc('in'),0,1),h)
        out = torch.concat([moy, hi,hj],dim=1)
        
        out = self.W1(out)
        out = self.relu(out)
        out = self.W2(out)
        return self.act(out)[:,0]

# ## Training loop
def print_logs(dataset_type: str, logs: dict):
    """Print the logs.

    Args
    ----
        dataset_type: Either "Train", "Eval", "Test" type.
        logs: Containing the metric's name and value.
    """
    desc = [
        f'{name}: {value:.2f}'
        for name, value in logs.items()
    ]
    desc = '\t'.join(desc)
    desc = f'{dataset_type} -\t' + desc
    desc = desc.expandtabs(5)
    # print(desc, end='\t| ')


def loss_batch(model, batched_graph, loss_fn):
    
    metrics = dict()

    node_features = batched_graph.ndata['features'].float()
    edge_features = batched_graph.edata['features'].float()
    edge_label = batched_graph.edata['label'].float()
    pred = model(batched_graph, node_features, edge_features)

    # Loss
        # pred (number_edges,2)
    mask_edges = batched_graph.edata['mask']
    metrics['loss'] = loss_fn(pred[mask_edges],edge_label[mask_edges])

    metrics['ROC'] = roc_auc_score(edge_label[mask_edges], pred[mask_edges].detach().numpy())

    return metrics


def eval_model(model, dataloader, loss_fn):
    logs = {'loss':[],'ROC':[]}
    
    with torch.no_grad():
        for batched_graph in dataloader:
            metrics = loss_batch(model, batched_graph, loss_fn)
            for name, value in metrics.items():
                logs[name].append(value.item())

    for name, values in logs.items():
        logs[name] = np.mean(values)
    return logs


def train_model(model, config):

    train_loader = config['train_loader']
    val_loader = config['val_loader'] 
    optimizer = config['optimizer']
    patience = config['patience']
    epochs = config['epochs']
    loss_fn = config['loss_fn']
    # Early stopping
    last_loss = 100
    trigger_times = 0

    for e in range(epochs):
        # print(f'\nEpoch {e+1}')
        model.train(True)      
        logs = {'loss':[],'ROC':[]}

        running_loss = 0
        for batch_id, batch in enumerate(train_loader):
            optimizer.zero_grad()
            metrics = loss_batch(model, batch, loss_fn)
            loss = metrics['loss']
            loss.backward()
            running_loss += loss
            optimizer.step()

            for name, value in metrics.items():
                logs[name].append(value.item())  # Don't forget the '.item' to free the cuda memory

            if batch_id % config['log_every'] == 0:
                for name, value in logs.items():
                    logs[name] = np.mean(value)

                train_logs = {
                    f'Train - {m}': v
                    for m, v in logs.items()
                }
                if config['wandb']:
                    wandb.log(train_logs)
                logs = {'loss':[],'ROC':[]}

        model.train(False)
        # Logs
        if len(logs) != 0:
            for name, value in logs.items():
                logs[name] = np.mean(value)
            train_logs = {
                f'Train - {m}': v
                for m, v in logs.items()
            }
        else:
            logs = {
                m.split(' - ')[1]: v
                for m, v in train_logs.items()
            if ' - ' in m}

        # print_logs('Train', logs)

        logs = eval_model(model, val_loader, loss_fn)
        # print_logs('Eval', logs)
        val_logs = {
            f'Validation - {m}': v
            for m, v in logs.items()
        }
        if config['wandb']:
            wb_logs = {**train_logs, **val_logs}  # Merge dictionnaries
            wandb.log(wb_logs)

        # Early stopping
        current_loss = logs['loss']
        if current_loss > last_loss:
            trigger_times += 1
            # print('Trigger Times:', trigger_times)
            if trigger_times >= patience:
                # print('Early stopping!')
                return model
        else:
            # print('Trigger Times: 0')
            trigger_times = 0
            last_loss = current_loss
    
    return model

def BCELoss_class_weighted(weights):

    def loss(input, target):
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        return torch.mean(bce)

    return loss



