import dgl
from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv, GATConv, GATv2Conv
from dgl.data.utils import split_dataset
import numpy as np

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class MDEVSPNodesDataParser:

    def __call__(self, df: pd.DataFrame):
        parsed = {}

        nodes_features = df[[c for c in df.columns if c in ['o', 'k', 'n', 'w', 'c', 'd', 'nb_dep_10', 'nb_dep_10_id', 'nb_fin_10', 'nb_fin_10_id', 't_s', 't_e'] or 's_' in c or 'e_' in c]].to_numpy()
        #nodes_features = df[[c for c in df.columns if c in ['pi_value']]].to_numpy()
        nodes_features = np.rint(nodes_features)
        parsed['feat'] = nodes_features

        df['mask'] = torch.zeros(len(df))
        df.loc[df['type'] == 'n', 'mask'] = 1
        dt = df['mask'].to_numpy().squeeze()
        parsed['mask'] = dt

        pi_vals = df['pi_value'].to_numpy().squeeze()
        parsed['pi_value'] = np.rint(pi_vals)

        df['cheat_class'] = torch.ones(len(df))
        df.loc[df['pi_value'] < 1, 'cheat_class'] = 0
        dt = torch.tensor(df['cheat_class'].values)
        
        dt = F.one_hot(dt.to(torch.int64), num_classes=2)
        

        parsed['label'] = dt

        return parsed

class MDEVSPEdgesDataParser:

    def __call__(self, df: pd.DataFrame):
        
        parsed = {}

        edges_features = df[['cost', 'energy', 'travel_time', 'waiting_time', 'delta_time', 'rg_id','rg']].to_numpy().squeeze()
        parsed['feat'] = edges_features

        return parsed

class BinaryClassifier(nn.Module):
    def __init__(self, in_size, hid_size, heads) -> None:
        super(BinaryClassifier, self).__init__()

        self.hid_size = hid_size


        self.gat1 = GATConv(in_size, hid_size, heads[0], activation=F.elu, allow_zero_in_degree=True)
        self.gat2 = GATConv(hid_size*heads[0], hid_size, heads[1], residual=True, activation=None, allow_zero_in_degree=True)
        #self.gat3 = GATConv(hid_size*heads[1], hid_size, heads[2], residual=True, activation=None, allow_zero_in_degree=True)

        #self.gat1 = GATConv(in_size, hid_size, heads[0], activation=F.elu, allow_zero_in_degree=True)
        #self.gat2 = GATConv(hid_size*heads[0], hid_size, heads[1], residual=True, activation=F.elu, allow_zero_in_degree=True)
        #self.gat_layers.append(GATConv(hid_size*heads[1], out_size, heads[2], residual=True, activation=None, allow_zero_in_degree=True))

        # 2 * hid_size car on cacatene
        #self.l1 = nn.Linear(2*hid_size, 32)
        #self.l2 = nn.Linear(32, 64)
        #self.l3 = nn.Linear(64, 1)
        
        # 2 * in_size because we stack two nodes 
        self.l1 = nn.Linear(2*hid_size, hid_size)
        self.l2 = nn.Linear(hid_size, hid_size)
        self.l3 = nn.Linear(hid_size, 1)

        #self.l1 = nn.Linear(2, 1)

        #self.pad = nn.ConstantPad2d((0, hid_size, 0, 0), 1)

    def forward(self, graph, inputs):

        h = inputs

        h = self.gat1(graph, h).flatten(1)
        h = self.gat2(graph, h).mean(1)



        # Concatene pour avoir les paires possibles
        num_nodes = graph.num_nodes()
        feats_size = h.shape[1]

        first = h.repeat(num_nodes,1)
        second = h.unsqueeze(1)
        # second = second.repeat(1,1,num_nodes)
        # second = second.view(num_nodes*num_nodes,-1,feats_size)
        # second = second.squeeze(1)
        second = second.repeat(1,1,num_nodes).view(num_nodes*num_nodes,-1,feats_size).squeeze(1)

        h = torch.cat((second,first), dim=1)
        # ===================== #

        h = torch.relu(self.l1(h))
        h = torch.relu(self.l2(h))
        h = torch.sigmoid(self.l3(h))

        return h.squeeze(1)

def evaluate(g, features, labels, mask, model):

    model.eval()
    with torch.no_grad():
        output = model(g, features)
        
        output = output[mask]
        labels = labels[mask]

        pred = np.where(output.data.cpu().numpy() >= 0, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), pred, average='micro')

        y_hat = output.argmax(1)
        y = labels.argmax(1)

        acc = accuracy_score(y_hat, y)

        return score, acc

def batch_loss(model, batched_graph, loss_fnc, second_greater_first, iter, both_are_trips):

    # Compute une prediction du modele, compute la loss et l'accuracy de la prediction, retourne loss et acc

    features = batched_graph.ndata['feat'].float()
    #mask = batched_graph.ndata['mask'].bool()

    #pi_values = batched_graph.ndata['pi_value'].float()
    
    probs = model(batched_graph, features)

    preds = (probs > 0.5).float()


    # if iter % 1 == 0:
    #     percent_1_real = torch.sum(second_greater_first) / second_greater_first.shape[0]
    #     percent_1_pred = torch.sum(preds) / preds.shape[0]

    #     print('         Real : {:.4f} | Preds : {:.4f}'.format(percent_1_real, percent_1_pred))

    #pi_values = pi_values[mask]
    #preds = preds[mask]

    

    goods = torch.eq(preds[both_are_trips], second_greater_first[both_are_trips]).float()
    acc = (torch.sum(goods) / goods.shape[0]).item()

    loss = loss_fnc(probs, second_greater_first)

    #preds = preds[mask]
    #pi_values = pi_values[mask]

    # acc = accuracy_score(second_greater_first, preds)
    
    return loss, acc

def evaluate_in_batches(dataloader, loss_fnc, device, model):
    
    total_loss = 0
    total_acc = 0

    with torch.no_grad():

        for batch_id, (batched_graph, _) in enumerate(dataloader):


            num_nodes = batched_graph.num_nodes()
            pi_values = batched_graph.ndata['pi_value'].unsqueeze(1)

            first = pi_values.repeat(num_nodes,1)
            second = pi_values.unsqueeze(1)
            second = second.repeat(1,1,num_nodes).view(num_nodes*num_nodes,-1,1).squeeze(1)

            second_minus_first = second - first

            second_greater_first = (second_minus_first > 0).float().squeeze(1)

            batched_graph = batched_graph.to(device)

            trips = batched_graph.ndata['mask'].unsqueeze(1)
            a = trips.repeat(num_nodes,1)
            b = trips.unsqueeze(1).repeat(1,1,num_nodes).view(num_nodes*num_nodes,-1,1).squeeze(1)
            both_are_trips = (a * b).squeeze(1)
            # percent_trips = torch.sum(both_are_trips) / both_are_trips.shape[0]
            # print(percent_trips)
            both_are_trips = both_are_trips.bool()
            
            loss, acc = batch_loss(model, batched_graph, loss_fnc, second_greater_first, batch_id, both_are_trips)

            total_loss += loss.item()
            total_acc += acc
            
            #features = batched_graph.ndata['feat']
            #labels = batched_graph.ndata['label']

            #mask = torch.tensor(batched_graph.ndata['mask'], dtype=torch.bool)

            #f1_acc, acc = evaluate(batched_graph, features, labels, mask, model)
            
            #total_f1_acc += f1_acc
            #total_acc += acc
        return (total_loss / (batch_id + 1)), (total_acc / (batch_id + 1))

def train(train_dataloader, val_dataloader, device, model):

    last_loss = 1000
    trigger_times = 0
    patience = 5

    loss_fcn = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)

    for epoch in range(400):
        
        model.train(True)

        train_total_loss = 0
        train_total_acc = 0
        # mini-batch loop
        for batch_id, (batched_graph, _) in enumerate(train_dataloader):
            batched_graph = batched_graph.to(device)
            # features = batched_graph.ndata['feat'].float()
            # mask = batched_graph.ndata['mask'].bool()

            # labels = batched_graph.ndata['label'].float()
            # logits = model(batched_graph, features)

            # loss = loss_fcn(logits[mask], labels[mask])

            num_nodes = batched_graph.num_nodes()
            pi_values = batched_graph.ndata['pi_value'].unsqueeze(1)

            first = pi_values.repeat(num_nodes,1)
            second = pi_values.unsqueeze(1)
            second = second.repeat(1,1,num_nodes).view(num_nodes*num_nodes,-1,1).squeeze(1)

            second_minus_first = second - first

            second_greater_first = (second_minus_first > 0).float().squeeze(1)

            trips = batched_graph.ndata['mask'].unsqueeze(1)
            a = trips.repeat(num_nodes,1)
            b = trips.unsqueeze(1).repeat(1,1,num_nodes).view(num_nodes*num_nodes,-1,1).squeeze(1)
            both_are_trips = (a * b).squeeze(1)

            #percent_trips = torch.sum(both_are_trips) / both_are_trips.shape[0]
            #print(percent_trips)

            both_are_trips = both_are_trips.bool()


            # Memes operations dans le meme sens donc ca vs ce qui sort du forward peuvent etre comparer
            
            loss, acc = batch_loss(model, batched_graph, loss_fcn, second_greater_first, batch_id, both_are_trips)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_total_loss += loss.item()
            train_total_acc += acc

            #f1_acc, acc = evaluate(batched_graph, features, labels, mask, model)

            if batch_id % 25 == 0:
                print('     Batch {:03d} | Loss : {:.4f} | Acc : {:.4f}'.format(batch_id, loss, acc))

        train_loss = train_total_loss / (batch_id + 1)
        train_acc = train_total_acc / (batch_id + 1)

        # Compute le validation acc et loss

        # for param in model.parameters():
        #     print(param)

        model.train(False)
        # On evalue le model avec le validation set : loss et acc

        eval_loss, eval_acc = evaluate_in_batches(val_dataloader, loss_fcn, device, model)

        print("Epoch {:05d} | Train Loss : {:.4f} | Eval Loss : {:.4f} | Train acc : {:.4f} | Eval Acc : {:.4f}".format(epoch, train_loss, eval_loss, train_acc, eval_acc))


        # if eval_loss > last_loss:
        #     trigger_times += 1
        #     if trigger_times >= patience:
        #         print('Early Stopping')
        #         return model
        # else:
        #     trigger_times = 0
        #     last_loss = eval_loss

    return model
    

if __name__ == '__main__':

    if torch.cuda.is_available():
        print('cuda is available')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    load = False
    ds = dgl.data.CSVDataset('./MDEVSP_dataset',ndata_parser=MDEVSPNodesDataParser(),edata_parser=MDEVSPEdgesDataParser(), force_reload=load)

    train_ds, val_ds, test_ds = split_dataset(ds, [0.8,0.1,0.1], shuffle=True)

    train_batch_size = 1
    val_test_batch_size = 1

    train_dataloader = GraphDataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_dataloader = GraphDataLoader(val_ds, batch_size=val_test_batch_size, shuffle=True)
    test_dataloader = GraphDataLoader(test_ds, batch_size=val_test_batch_size, shuffle=True)

    g,_ = train_ds[0]
    features = g.ndata['feat']

    in_size = features.shape[1]
    model = BinaryClassifier(in_size, 32, [3, 3])

    # model training

    print('Training ...')
    best_model = train(train_dataloader, val_dataloader, device, model)

    print('Testing...')
    test_loss_fnc = nn.BCEWithLogitsLoss()
    test_loss, test_acc = evaluate_in_batches(test_dataloader, test_loss_fnc, device, model)
    print("Test Loss {:.4f} | Test Acc {:.4f}".format(test_loss, test_acc))


