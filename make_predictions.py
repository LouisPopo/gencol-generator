# get a model and make predictions for ALL instances.

import torch
import dgl
from dgl.dataloading import GraphDataLoader
import pickle
import pandas as pd
import os
import subprocess

from train_model_CLASSIFIER import BinaryClassifier, MDEVSPEdgesDataParser, MDEVSPNodesDataParser

if torch.cuda.is_available():
        print('cuda is available')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('Networks/instances_id_to_info.pkl', 'rb') as f:
    graph_id_to_instance = pickle.load(f)

df_nodes = pd.read_csv('MDEVSP_dataset/nodes.csv')
df_nodes_graphs_infos = df_nodes[['name', 'node_id', 'graph_id']]

ds = dgl.data.CSVDataset('./MDEVSP_dataset',ndata_parser=MDEVSPNodesDataParser(),edata_parser=MDEVSPEdgesDataParser(), force_reload=False)

all_dataloader = GraphDataLoader(ds, batch_size=1, shuffle=True)

g,_ = ds[0]
nodes_features = g.ndata['feat']
edges_features = g.edata['feat']
nodes_in_size = nodes_features.shape[1]
edges_in_size = edges_features.shape[1]

model_name = '  '

model = BinaryClassifier(nodes_in_size, edges_in_size, [(nodes_in_size,edges_in_size), (32,0)], [3, 3, 3, 6], 15)

model.load_state_dict(torch.load('models/{}/model.pt'.format(model_name)))

model.to(DEVICE)

total_nb_instances = len(ds)
nb_done = 0

for batch_id, (batched_graph, data) in enumerate(all_dataloader):
    batched_graphs = dgl.unbatch(batched_graph)
    nb_graphs = len(batched_graphs)

    batch_losses = []
    batch_accs = []

    for batched_graph in batched_graphs:

        # Make the gencol file
        graph_id = data['id'].item()
        graph_instance = graph_id_to_instance[graph_id]
        instance_folder = 'Network{}'.format(graph_instance)

        file_path = 'Networks/Network{}/inequalities_predictions.csv'.format(graph_instance)

        # done = False

        # if os.path.isfile(file_path):
        #     date = subprocess.check_output(['date', '+%D', '-r', file_path]).decode('utf-8')
            
        #     day = int(date.split('/')[1])
        #     if day >= 21:
        #         done = True
        #         nb_done += 1
        
        # if done:
        #     continue
            
        # continue

        batched_graph = dgl.add_self_loop(batched_graph)

        batched_graph = batched_graph.to(DEVICE)

        is_trip = batched_graph.ndata['mask'].bool()
        num_trip_nodes = torch.sum(is_trip)

        pi_values = batched_graph.ndata['pi_value'].unsqueeze(1)[is_trip]

        B_pi = pi_values.repeat(num_trip_nodes,1)
        A_pi = pi_values.unsqueeze(1).repeat(1,1,num_trip_nodes).view(num_trip_nodes*num_trip_nodes,-1,1).squeeze(1)

        second_minus_first = A_pi - B_pi

        second_greater_first = (second_minus_first > 0).float().squeeze(1)

        second_greater_first.to(DEVICE)

        features = batched_graph.ndata['feat'].float()

        features = features.to(DEVICE)

        probs = model(batched_graph, features)

        preds = (probs > 0.5).float()

        # Model accuracy
        goods = torch.eq(preds, second_greater_first).float()
        acc = (torch.sum(goods) / goods.shape[0]).item()



        

        df_nodes_graph = df_nodes_graphs_infos[df_nodes_graphs_infos["graph_id"] == graph_id]

        nodes_ids = df_nodes_graphs_infos.loc[
            df_nodes_graphs_infos['graph_id'] == graph_id, "node_id"
        ]

        trip_nodes_ids = torch.tensor(nodes_ids[is_trip.tolist()].values)
        
        trip_nodes_ids.to(DEVICE)

        trip_nodes_ids = torch.reshape(trip_nodes_ids, (-1,1))
        first = trip_nodes_ids.repeat(num_trip_nodes, 1)
        second = trip_nodes_ids.unsqueeze(1).repeat(1,1,num_trip_nodes).view(num_trip_nodes*num_trip_nodes,-1,1).squeeze(1)

        probs = probs.unsqueeze(1).cpu()
        # preds = preds.unsqueeze(1).to(DEVICE)
        second_greater_first = second_greater_first.unsqueeze(1)

        # preds = preds.cpu()
        #second_greater_first = second_greater_first.cpu()
        second_greater_first = second_greater_first.cpu()
        A_pi = A_pi.cpu()
        B_pi = B_pi.cpu()

        results = torch.cat((second, first, probs, second_greater_first, A_pi, B_pi), dim=1)

        df_results = pd.DataFrame(results.detach().numpy(), columns=['A', 'B', 'pred', 'real', 'A_pi', 'B_pi'])

        node_id_to_name_dict = dict(zip(df_nodes_graph.node_id, df_nodes_graph.name))

        df_results_with_names = df_results.replace({'A' : node_id_to_name_dict, 'B' : node_id_to_name_dict})

        df_results_with_names.to_csv('Networks/{}/inequalities_predictions.csv'.format(instance_folder))

    print(graph_instance)
    print('{}/{} done'.format(batch_id+1, total_nb_instances))
    
print(nb_done)