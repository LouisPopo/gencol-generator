import enum
from random import shuffle
from unicodedata import name
import dgl
import torch

from dgl.dataloading import GraphDataLoader

import pandas as pd

import networkx as nx
import torch.nn as nn

import numpy as np

import torch.nn.functional as F

from dgl.data import DGLDataset
from dgl.nn import GraphConv

class MDEVSPDataset(DGLDataset):

    def __init__(self, list_graphs):
        self.list_graphs = list_graphs
        super().__init__(name='MDEVSP')

    def process(self):

        self.graphs = []

        for pb in self.list_graphs:

            nodes_data = pd.read_csv('Networks/Network{}/graph_nodes.csv'.format(pb),sep=';',index_col = 0)
            edges_data = pd.read_csv('Networks/Network{}/graph_edges.csv'.format(pb),sep=';',index_col = 0)
        
            edges_features = torch.tensor(edges_data[['cost', 'energy', 'travel_time', 'waiting_time', 'delta_time', 'rg_id','rg']].astype('float').values, dtype=torch.float)
            nodes_features = torch.tensor(nodes_data[[c for c in nodes_data.columns if c in ['o', 'k', 'n', 'w', 'c', 'd', 'nb_dep_10', 'nb_dep_10_id', 'nb_fin_10', 'nb_fin_10_id', 't_s', 't_e', 'pi_value'] or 's_' in c or 'e_' in c]].astype('float').values, dtype=torch.float)

            # node_labels = torch.tensor(nodes_data['class'].astype('category').cat.codes.to_numpy())
            node_labels = torch.tensor(nodes_data['class'].astype('float').values, dtype=torch.long)

            graph = dgl.graph((edges_data['idx_src'], edges_data['idx_dst']))

            #self.graph = dgl.add_self_loop(self.graph)

            graph.ndata['feat'] = nodes_features
            graph.ndata['label'] = node_labels
            graph.edata['feat'] = edges_features

            # n_nodes = nodes_data.shape[0]
            # n_train = int(n_nodes*0.6)
            # n_val = int(n_nodes*0.2)

            # train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            # val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            # test_mask = torch.zeros(n_nodes, dtype=torch.bool)

            # train_mask[:n_train] = True
            # val_mask[n_train:n_train+n_val] = True
            # test_mask[n_train+n_val:] = True

            # df_mask = pd.DataFrame()
            # df_mask['train_mask'] = train_mask
            # df_mask['val_mask'] = val_mask
            # df_mask['test_mask'] = test_mask

            # df_mask = df_mask.sample(frac=1).reset_index(drop=True)

            # graph.ndata['train_mask'] = torch.tensor(df_mask['train_mask'])
            # graph.ndata['val_mask'] = torch.tensor(df_mask['val_mask'])
            # graph.ndata['test_mask'] = torch.tensor(df_mask['test_mask'])

            self.graphs.append(graph)

            print('Graph for {} is done'.format(pb))

        self.num_classes = 2

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)

class GCN(nn.Module):

    def __init__(self, in_feats, h_feats, num_classes) -> None:
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, num_classes, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


def eval_model(model, dataloader):

    val_accs = []

    with torch.no_grad():
        for g in dataloader:

            features = g.ndata['feat'].float()
            labels = g.ndata['label']

            logits = model(g, features)

            pred = logits.argmax(1)

            loss = F.cross_entropy(logits, labels)

            val_acc = (pred == labels).float().mean()
            
            val_accs.append(val_acc)

    return np.mean(val_accs)


from glob import glob
import os

list_graphs = []

for instance_folder in glob('Networks/Network*'):
    instance_info = instance_folder.split('/')[1].replace('Network', '')

    if os.path.exists('{}/reportProblem{}_default.out'.format(instance_folder, instance_info)):

        list_graphs.append(instance_info)

np.random.shuffle(list_graphs)

nb_train = int(0.80*len(list_graphs))
nb_valid = int(0.10*len(list_graphs))
nb_test = len(list_graphs) - nb_train - nb_valid

train, validation, test = np.split(list_graphs, [nb_train, nb_train + nb_valid])

print(train)
print(validation)
print(test)

dataloader_train = GraphDataLoader(MDEVSPDataset(train), shuffle=True)
dataloader_eval = GraphDataLoader(MDEVSPDataset(validation), shuffle=True)
dataloader_test = GraphDataLoader(MDEVSPDataset(test), shuffle=True)


g = MDEVSPDataset(test)[0]


node_features = g.ndata['feat'].float()
edges_features = g.edata['feat'].float()
node_label = g.ndata['label']

model = GCN(g.ndata['feat'].shape[1], 16, 2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
best_val_acc = 0
best_test_acc = 0

print('Starting training')

epochs = 150

for e in range(epochs):

    model.train(True)

    train_accs = []

    for id, graph in enumerate(dataloader_train): # on train sur 5 graphs, on ajoute les parameters

        features = graph.ndata['feat'].float()
        labels = graph.ndata['label']

        optimizer.zero_grad()
        logits = model(graph, features)

        pred = logits.argmax(1)

        loss = F.cross_entropy(logits, labels)

        loss.backward()

        t = (pred == labels).float().mean() # SUR UN SEUL GRAPHE

        train_accs.append(t)

        optimizer.step()

    train_acc = np.mean(train_accs)

    model.train(False)

    val_acc = eval_model(model, dataloader_eval)
    test_acc = eval_model(model, dataloader_test)

    if best_val_acc < val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc

    if e % 5 == 0:
        print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
            e, train_acc, val_acc, best_val_acc, test_acc, best_test_acc))


  