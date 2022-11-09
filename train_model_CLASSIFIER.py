import math
import random
import dgl
import pickle
import json
from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv, GATConv, GATv2Conv, EGATConv
from dgl.data.utils import split_dataset
import numpy as np

import os
from datetime import datetime

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = None

pd.options.mode.chained_assignment = None

class MDEVSPNodesDataParser:

    def __call__(self, df: pd.DataFrame):
        parsed = {}

        # Data normalization 3.2

        t_min = df.loc[df['t_s'] > 0, 't_s'].min()
        df.loc[df['o'] == 1, 't_s'] = t_min
        df.loc[df['o'] == 1, 't_e'] = t_min
        df['duration'] = df['t_e'] - df['t_s']
        df['duration'] = df['duration']/df['duration'].max()
        
        df['t_s'] = (df['t_s'] - t_min) / (df['t_s'].max() - t_min)
        df['t_e'] = (df['t_e'] - t_min) / (df['t_e'].max() - t_min)

        df['nb_dep_10'] = (df['nb_dep_10'] - df['nb_dep_10'].min()) / (df['nb_dep_10'].max() - df['nb_dep_10'].min())
        df['nb_dep_id_10'] = (df['nb_dep_id_10'] - df['nb_dep_id_10'].min()) / (df['nb_dep_id_10'].max() - df['nb_dep_id_10'].min())
        df['nb_fin_10'] = (df['nb_fin_10'] - df['nb_fin_10'].min()) / (df['nb_fin_10'].max() - df['nb_fin_10'].min())
        df['nb_fin_id_10'] = (df['nb_fin_id_10'] - df['nb_fin_id_10'].min()) / (df['nb_fin_id_10'].max() - df['nb_fin_id_10'].min())

        # ==================
        nodes_features = df[[c for c in df.columns if c in ['o', 'k', 'n', 'w', 'c', 'd', 'nb_dep_10', 'nb_dep_id_10', 'nb_fin_10', 'nb_fin_id_10', 't_s', 't_e'] or 's_' in c or 'e_' in c]].to_numpy()

        #nodes_features = df[[c for c in df.columns if c in ['pi_value']]].to_numpy()
        
        # nodes_ids = df['node_id'].to_numpy()
        # parsed['ids'] = nodes_ids

        # 3.0 -> 3.1 (Quand on normalise pas)
        #nodes_features = np.rint(nodes_features)

        #node_names = df['name'].to_numpy()
        #parsed['name'] = node_names

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
        
        # Data Normalization

        # Min = 0 donc...

        df['cost'] = df['cost'] / df['cost'].max()
        df['energy'] = df['energy'] / df['energy'].max()
        for t in ['travel', 'waiting', 'delta']:
            df['{}_time'.format(t)] = df['{}_time'.format(t)] / df['{}_time'.format(t)].max()

        # ======== 
        edges_features = df[['cost', 'energy', 'travel_time', 'waiting_time', 'delta_time', 'rg_id','rg']].to_numpy().squeeze()

        parsed['feat'] = edges_features

        return parsed

class BinaryClassifier(nn.Module):
    def __init__(self, nodes_in_size, edges_in_size, hid_size, heads) -> None:
        super(BinaryClassifier, self).__init__()

        self.hid_size = hid_size
        
        self.egat1 = EGATConv(nodes_in_size, edges_in_size, hid_size, hid_size, heads[0])
        self.egat2 = EGATConv(hid_size*heads[0], hid_size*heads[0], hid_size, hid_size, heads[1])
        self.egat3 = EGATConv(hid_size*heads[1], hid_size*heads[1], hid_size, hid_size, heads[2])
        
        # v3_8 : 
        self.ml1 = nn.Linear((nodes_in_size + hid_size), 2*hid_size)
        # =====
        #self.ml1 = nn.Linear(hid_size, 2*hid_size)
        self.ml2 = nn.Linear(2*hid_size, 2*hid_size)
        self.ml3 = nn.Linear(2*hid_size, 2*hid_size)
        self.ml4 = nn.Linear(2*hid_size, hid_size)
        
        # Concat
        self.l1 = nn.Linear(2*hid_size, 32)
        self.l2 = nn.Linear(32, 64)
        self.l3 = nn.Linear(64, 128)
        self.l4 = nn.Linear(128, 128)
        self.l5 = nn.Linear(128, 64)
        self.l6 = nn.Linear(64, 32)
        self.l7 = nn.Linear(32, 16)
        self.l8 = nn.Linear(16, 1)

        #self.l1 = nn.Linear(2, 1)

        #self.pad = nn.ConstantPad2d((0, hid_size, 0, 0), 1)

    def forward(self, graph, inputs):

        is_trip = graph.ndata['mask'].bool()
        num_trip_nodes = torch.sum(is_trip)

        nodes_feats = inputs
        edges_feats = graph.edata['feat'] 

        nodes_feats.to(DEVICE)
        edges_feats.to(DEVICE)

        # PAS DE SKIP CONNECTION EN CE MOMENT

        # 1. GNN
        #h = inputs

        # v3_2
        #nodes_feats, edges_feats = self.gat1(graph, nodes_feats, edges_feats)
        #h = nodes_feats.mean(1)
        # edges_feats = edges_feats.mean(1)

        # v3_4

        # nodes_feats, edges_feats = self.egat1(graph, nodes_feats, edges_feats)
        # nodes_feats = nodes_feats.flatten(1)
        # edges_feats = edges_feats.flatten(1)

        # nodes_feats, edges_feats = self.egat2(graph, nodes_feats, edges_feats)
        # nodes_feats = nodes_feats.flatten(1)
        # edges_feats = edges_feats.flatten(1)

        # nodes_feats, edges_feats = self.egat3(graph, nodes_feats, edges_feats)
        # nodes_feats = nodes_feats.mean(1)
        # edges_feats = edges_feats.mean(1)

        # v3_5
        # nodes_feats, edges_feats = self.egat1(graph, nodes_feats, edges_feats)
        # nodes_feats = torch.max(nodes_feats, 1).values
        # edges_feats = torch.max(edges_feats, 1).values

        # nodes_feats, edges_feats = self.egat2(graph, nodes_feats, edges_feats)
        # nodes_feats = torch.max(nodes_feats, 1).values
        # edges_feats = torch.max(edges_feats, 1).values

        # nodes_feats, edges_feats = self.egat3(graph, nodes_feats, edges_feats)
        # nodes_feats = torch.max(nodes_feats, 1).values
        # edges_feats = torch.max(edges_feats, 1).values

        # v3_7

        # nodes_feats, edges_feats = self.egat1(graph, nodes_feats, edges_feats)
        # nodes_feats = nodes_feats.flatten(1)
        # edges_feats = edges_feats.flatten(1)

        # nodes_feats, edges_feats = self.egat2(graph, nodes_feats, edges_feats)
        # nodes_feats = nodes_feats.flatten(1)
        # edges_feats = edges_feats.flatten(1)

        # nodes_feats, edges_feats = self.egat3(graph, nodes_feats, edges_feats)
        # nodes_feats = torch.max(nodes_feats, 1).values
        # edges_feats = torch.max(edges_feats, 1).values

        # v3_8
        nodes_feats, edges_feats = self.egat1(graph, nodes_feats, edges_feats)
        nodes_feats = nodes_feats.flatten(1)
        edges_feats = edges_feats.flatten(1)

        # print(nodes_feats.is_cuda)

        nodes_feats, edges_feats = self.egat2(graph, nodes_feats, edges_feats)
        nodes_feats = nodes_feats.flatten(1)
        edges_feats = edges_feats.flatten(1)

        nodes_feats, edges_feats = self.egat3(graph, nodes_feats, edges_feats)
        nodes_feats = nodes_feats.mean(1)
        edges_feats = edges_feats.mean(1)


        h = nodes_feats

        # v3_8 : concat nodes_features + inputs

        h = torch.cat((h, inputs), 1)

        # h = self.gat1(graph, h).flatten(1)
        # h = self.gat2(graph, h).flatten(1)
        # h = self.gat3(graph, h).flatten(1)
        # h = self.gat4(graph, h).mean(1)

        # Si on passe dans le mid mlp
        # 2. MID MLP (pour réduire le nb. de features)
        h = h[is_trip]
        h = torch.relu(self.ml1(h))
        h = torch.relu(self.ml2(h))
        h = torch.relu(self.ml3(h))
        h = self.ml4(h) # PAS DE RELU COMME CA ON A UNE VALEUR PAS CONTRAINTES! 

        # 3. Concatene ensemble
        feats_size = h.shape[1]
        first = h.repeat(num_trip_nodes, 1)
        second = h.unsqueeze(1).repeat(1,1,num_trip_nodes).view(num_trip_nodes*num_trip_nodes,-1,feats_size).squeeze(1)
        h = torch.cat((second, first), dim=1)

        # 4. MLP pour prédiction 
        h = torch.relu(self.l1(h))
        h = torch.relu(self.l2(h))
        h = torch.relu(self.l3(h))
        h = torch.relu(self.l4(h))
        h = torch.relu(self.l5(h))
        h = torch.relu(self.l6(h))
        h = torch.relu(self.l7(h))
        h = torch.sigmoid(self.l8(h))


        # Version juste avec les nodes trips
        
        # ===================== #

        # Version avec tous les noeuds
        # num_nodes = graph.num_nodes()
        # feats_size = h.shape[1]
        # first = h.repeat(num_nodes,1)
        # second = h.unsqueeze(1).repeat(1,1,num_nodes).view(num_nodes*num_nodes,-1,feats_size).squeeze(1)
        # h = torch.cat((second,first), dim=1)
        # ===================== #

        # h = torch.relu(self.l1(h))
        # h = torch.relu(self.l2(h))
        # h = torch.sigmoid(self.l3(h))

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

def batch_loss(model, batched_graph, loss_fnc, second_greater_first, iter):

    # Compute une prediction du modele, compute la loss et l'accuracy de la prediction, retourne loss et acc

    features = batched_graph.ndata['feat'].float()
    #mask = batched_graph.ndata['mask'].bool()

    #pi_values = batched_graph.ndata['pi_value'].float()

    #print('   IN BATCH LOSS : batched_graph : {}'.format(batched_graph.get_device()))
    # print('   IN BATCH LOSS : features : {}'.format(features.get_device()))
    
    probs = model(batched_graph, features)

    preds = (probs > 0.5).float()


    # if iter % 1 == 0:
    #     percent_1_real = torch.sum(second_greater_first) / second_greater_first.shape[0]
    #     percent_1_pred = torch.sum(preds) / preds.shape[0]

    #     print('         Real : {:.4f} | Preds : {:.4f}'.format(percent_1_real, percent_1_pred))

    #pi_values = pi_values[mask]
    #preds = preds[mask]

    

    goods = torch.eq(preds, second_greater_first).float()
    acc = (torch.sum(goods) / goods.shape[0]).item()

    loss = loss_fnc(probs, second_greater_first)

    #preds = preds[mask]
    #pi_values = pi_values[mask]

    # acc = accuracy_score(second_greater_first, preds)
    
    return loss, acc, probs


def validate_logic_predictions(preds, second_greater_first):

    # from the graph we sample a few pair A,B
    # if A,B = 1 (A >= B) -> B,A = 1 (if B=A) or 0 (if B < A)
    # if B,A = 0 (A  < B) -> B,A = 1 (B>=A)
    n_nodes = int(math.sqrt(preds.shape[0]))
    n_pairs = preds.shape[0]
    #n_samples = 1000
    #samples = random.sample(range(0, n_pairs), n_samples)

    logic_respected = 0
    cnt = 0

    for i in range(0,n_nodes):
        for j in range(i + 1, n_nodes):

            cnt += 1

            idx_ab = i*n_nodes + j

            i_ = (idx_ab + 1) // n_nodes
            j_ = (idx_ab % n_nodes) + 1

            idx_ba = i_*n_nodes + j_

            p_ab = preds[idx_ab]
            p_ba = preds[idx_ba]

            r_ab = second_greater_first[idx_ab]
            r_ba = second_greater_first[idx_ba]

            if p_ab == 1:
                # A >= B then B >= A (1 or 0) is right if 1 : A == B, else A > B
                logic_respected += 1
            else:
                # A < B then B >= A (1)
                if p_ba == 1:
                    logic_respected += 1

        return logic_respected/cnt


    # logic_respected = 0

    # for s_ab in samples:

    #     p_ab = preds[s_ab].item() # notre prediction A,B (1 ou 0)

    #     i = (s_ab + 1) // n_nodes
    #     j = (s_ab % n_nodes) + 1

    #     s_ba = (j - 1)*n_nodes + i - 1
    #     p_ba = preds[s_ba].item()

    #     if p_ab == 1:
            
    #         # A >= B
    #         r_ab = second_greater_first[s_ab]
    #         r_ba = second_greater_first[s_ba]
    #         if r_ab == r_ba:
    #             # A == B alors B >= A : 1
    #             if p_ba == 1:
    #                 logic_respected += 1
    #         else:
    #             # A != B alors B >= A : 0
    #             if p_ba == 0:
    #                 logic_respected += 1
    #     else:

    #         # A < B
    #         if p_ba == 1:
    #             # B >= A
    #             logic_respected += 1

    # return logic_respected/n_samples
    

def evaluate_in_batches(dataloader, loss_fnc, device, model, df_nodes_graphs_infos=None, graph_id_to_instance=None, print_predictions=False):
    
    # df_nodes_graphs_infos : df avec le nom du noeud, son id et le graph id correspondant
    #                         df[['name', 'node_id', 'graph_id']]
    # graph_id_to_instance : dict qui donne le nom de l'instance en fonction du graph id
    # print_predictions : si on veut printer les predictions
     
    total_loss = 0
    total_acc = 0
    percent_logic_respected = []

    with torch.no_grad():

        for batch_id, (batched_graph, data) in enumerate(dataloader):

            # Ici, second greather first pourrait etre fait uniquement avec les trips nodes, pour eviter le deuxieme filtre

            # num_nodes = batched_graph.num_nodes()


            batched_graph = dgl.add_self_loop(batched_graph)

            is_trip = batched_graph.ndata['mask'].bool()
            num_trip_nodes = torch.sum(is_trip)

            pi_values = batched_graph.ndata['pi_value'].unsqueeze(1)[is_trip]

            first = pi_values.repeat(num_trip_nodes,1)
            second = pi_values.unsqueeze(1)
            second = second.repeat(1,1,num_trip_nodes).view(num_trip_nodes*num_trip_nodes,-1,1).squeeze(1)

            second_minus_first = second - first
            second_greater_first = (second_minus_first >= 0).float().squeeze(1)

            second_greater_first = second_greater_first.to(DEVICE)

            batched_graph = batched_graph.to(DEVICE)

            #batched_graph = batched_graph.to(device)

            #trips = batched_graph.ndata['mask'].unsqueeze(1)
            #a = trips.repeat(num_nodes,1)
            #b = trips.unsqueeze(1).repeat(1,1,num_nodes).view(num_nodes*num_nodes,-1,1).squeeze(1)
            #both_are_trips = (a * b).squeeze(1)
            # percent_trips = torch.sum(both_are_trips) / both_are_trips.shape[0]
            # print(percent_trips)
            #both_are_trips = both_are_trips.bool()

            # print('Batched graph : {}'.format(batched_graph.get_device()))
            #print('Second greater first : {}'.format(second_greater_first.get_device()))
            
            loss, acc, probs = batch_loss(model, batched_graph, loss_fnc, second_greater_first, batch_id)

            preds = (probs > 0.5).float()

            # tests (preds, second greater)

            log_res = validate_logic_predictions(preds, second_greater_first)
            percent_logic_respected.append(log_res)

            total_loss += loss.item()
            total_acc += acc
            
             # On va chercher le id (instance) du graph, apres on pourra l'utiliser. 
            if print_predictions:
                graph_id = data['id'].item()
                graph_instance = graph_id_to_instance[graph_id]
                instance_folder = 'Network{}'.format(graph_instance)
                
                df_nodes_graph = df_nodes_graphs_infos[df_nodes_graphs_infos["graph_id"] == graph_id]

                nodes_ids = df_nodes_graphs_infos.loc[
                    df['graph_id'] == graph_id, "node_id"
                ]

                trip_nodes_ids = torch.tensor(nodes_ids[is_trip.tolist()].values)
                
                trip_nodes_ids = torch.reshape(trip_nodes_ids, (-1,1))
                first = trip_nodes_ids.repeat(num_trip_nodes, 1)
                second = trip_nodes_ids.unsqueeze(1).repeat(1,1,num_trip_nodes).view(num_trip_nodes*num_trip_nodes,-1,1).squeeze(1)
                all_pair_nodes_ids = torch.cat((second, first), dim=1)

                preds = preds.unsqueeze(1)
                second_greater_first = second_greater_first.unsqueeze(1)

                results = torch.cat((second, first, preds, second_greater_first), dim=1)

                df_results = pd.DataFrame(results.numpy(), columns=['A', 'B', 'pred', 'real'])

                node_id_to_name_dict = dict(zip(df_nodes_graph.node_id, df_nodes_graph.name))

                df_results_with_names = df_results.replace({'A' : node_id_to_name_dict, 'B' : node_id_to_name_dict})

                df_results_with_names.to_csv('Networks/{}/inequalities_predictions.csv'.format(instance_folder))


        return (total_loss / (batch_id + 1)), (total_acc / (batch_id + 1)), np.average(percent_logic_respected)

def train(train_dataloader, val_dataloader, device, model, df_nodes_graphs_infos, graph_id_to_instance):

    last_loss = 1000
    trigger_times = 0 
    patience = 10

    loss_fcn = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    for epoch in range(1):
        
        model.train(True)

        train_total_loss = 0
        train_total_acc = 0
        # mini-batch loop
        for batch_id, (batched_graph, data) in enumerate(train_dataloader):
            
            
            batched_graph = dgl.add_self_loop(batched_graph)

            batched_graph = batched_graph.to(DEVICE)

            is_trip = batched_graph.ndata['mask'].bool()
            num_trip_nodes = torch.sum(is_trip)
            # features = batched_graph.ndata['feat'].float()
            # mask = batched_graph.ndata['mask'].bool()

            # labels = batched_graph.ndata['label'].float()
            # logits = model(batched_graph, features)

            # loss = loss_fcn(logits[mask], labels[mask])

            #num_nodes = batched_graph.num_nodes()
            pi_values = batched_graph.ndata['pi_value'].unsqueeze(1)[is_trip]

            first = pi_values.repeat(num_trip_nodes,1)
            second = pi_values.unsqueeze(1).repeat(1,1,num_trip_nodes).view(num_trip_nodes*num_trip_nodes,-1,1).squeeze(1)

            second_minus_first = second - first

            second_greater_first = (second_minus_first > 0).float().squeeze(1)

            second_greater_first.to(DEVICE)

            # trips = batched_graph.ndata['mask'].unsqueeze(1)
            # a = trips.repeat(num_nodes,1)
            # b = trips.unsqueeze(1).repeat(1,1,num_nodes).view(num_nodes*num_nodes,-1,1).squeeze(1)
            # both_are_trips = (a * b).squeeze(1)

            #percent_trips = torch.sum(both_are_trips) / both_are_trips.shape[0]
            #print(percent_trips)

            # both_are_trips = both_are_trips.bool()


            # Memes operations dans le meme sens donc ca vs ce qui sort du forward peuvent etre comparer
            
            loss, acc, probs = batch_loss(model, batched_graph, loss_fcn, second_greater_first, batch_id)

            preds = (probs > 0.5).float()
            
            # real_prcnt = torch.sum(second_greater_first, 0) / second_greater_first.shape[0]
            # pred_prcnt = torch.sum(preds, 0) / preds.shape[0]
            # print('Real % : {:.4f} | Preds % : {:.4f}'.format(real_prcnt, pred_prcnt))

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

        eval_loss, eval_acc, perc_logic_res = evaluate_in_batches(val_dataloader, loss_fcn, device, model)

        print("Epoch {:05d} | Train Loss : {:.4f} | Eval Loss : {:.4f} | Train acc : {:.4f} | Eval Acc : {:.4f} | Eval Log. Respected : {:.4f}".format(epoch, train_loss, eval_loss, train_acc, eval_acc, perc_logic_res))


        if eval_loss > last_loss:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early Stopping')
                return model
        else:
            trigger_times = 0
            last_loss = eval_loss

    return model
    

if __name__ == '__main__':

    # qqupart on doit lire nodes.csv :
    # Avoir un lien entre le graph_id, node_id et le name
    # Comme ca plus tard quand on filtre avec les mask, on peut filtre sur cette structure de données aussi. 


    if torch.cuda.is_available():
        print('cuda is available')

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('USING : {}'.format(DEVICE))

    load = True
    ds = dgl.data.CSVDataset('./MDEVSP_dataset',ndata_parser=MDEVSPNodesDataParser(),edata_parser=MDEVSPEdgesDataParser(), force_reload=load)

    train_ds, val_ds, test_ds = split_dataset(ds, [0.8,0.1,0.1], shuffle=True)

    train_batch_size = 1
    val_test_batch_size = 1

    train_dataloader = GraphDataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_dataloader = GraphDataLoader(val_ds, batch_size=val_test_batch_size, shuffle=True)
    test_dataloader = GraphDataLoader(test_ds, batch_size=val_test_batch_size, shuffle=True)

    g,_ = train_ds[0]
    nodes_features = g.ndata['feat']
    edges_features = g.edata['feat']
    

    nodes_in_size = nodes_features.shape[1]
    edges_in_size = edges_features.shape[1]
    model = BinaryClassifier(nodes_in_size, edges_in_size, 32, [3, 3, 3, 6])

    model.to(DEVICE)


    # MANIPS POUR FAIRE LE LIEN ENTRE GRAPH_ID,NODE_ID -> INSTANCE_INFO, NODE_NAME
    df = pd.read_csv('MDEVSP_dataset/nodes.csv')
    df_nodes_graphs_infos = df[['name', 'node_id', 'graph_id']]

    with open('Networks/instances_id_to_info.pkl', 'rb') as f:
        graph_id_to_instance = pickle.load(f)
    #

    test_loss_fnc = nn.BCEWithLogitsLoss()
    #test_loss, test_acc, perc_logic_res = evaluate_in_batches(test_dataloader, test_loss_fnc, DEVICE, model, df_nodes_graphs_infos, graph_id_to_instance, print_predictions=True)

    # model training

    print('Training ...')
    best_model = train(train_dataloader, val_dataloader, DEVICE, model, df_nodes_graphs_infos, graph_id_to_instance)

    # Save the model :
    
    dt_string = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    model_folder = 'models/{}/'.format(dt_string)
    model_path = '{}model.pt'.format(model_folder)
    
    os.mkdir(model_folder)
    torch.save(model.state_dict(), model_path)

    print('Testing...')
    # Ici on voudrait générer un fichier qui nous dit pour chaque graphes, chaque paires de noeuds, la relation prédite.
    # Avec ça on pourrait tester une résolution. 
    test_loss_fnc = nn.BCEWithLogitsLoss()
    test_loss, test_acc, perc_logic_res = evaluate_in_batches(test_dataloader, test_loss_fnc, DEVICE, model, df_nodes_graphs_infos, graph_id_to_instance, print_predictions=True)
    print("Test Loss {:.4f} | Test Acc {:.4f} | Test Log. Res. {:.4f}".format(test_loss, test_acc, perc_logic_res))


