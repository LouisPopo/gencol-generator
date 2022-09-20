from dbm import dumb
import enum
import dgl
import torch

import matplotlib.pyplot as plt
import networkx as nx

# Choose a network , instance and creates a DGL GRAPH withit : nodes + edges

# NODES IDS : 
#   Depot : o_D{n} , k_D{n} for n in range(nb_depots)
#   Trips : n_T{n}          for n in range(nb_trips)
#   Charging : w_H{n}_P{t}  for n in range(nb_bornes_recharge), for t in range(nb_periods)

# IMPORTANT D'AVOIR LES MÊMES

instance = "4b_4_0"

nodes_string_ids = dict()
nodes_int_ids = []

depot_file = open("Networks/Network{}/depots.txt".format(instance), "r")
depot_list = depot_file.readlines()

trips_file = open("Networks/Network{}/voyages.txt".format(instance), "r")
trips_list = trips_file.readlines()

print('Nb depots : {}'.format(len(depot_list)))
print('Nb trips : {}'.format(len(trips_list)))


list_of_nodes_features =[]

# Get les infos des depot
for i,depot in enumerate(depot_list):

    d = depot.split(';')

    n200 = int(d[1])

    o_str_id = 'o_D{}'.format(i)
    o_int_id = len(nodes_int_ids)
    nodes_int_ids.append(o_str_id)

    d_str_id = 'k_D{}'.format(i)
    d_int_id = len(nodes_int_ids)
    nodes_int_ids.append(d_str_id)

    nodes_string_ids[o_str_id] = {
        'type' : 'origin_node',
        'features' : { 'n_200' : n200 },
        'int_id' : o_int_id
    }

    list_of_nodes_features.append(torch.tensor([0,0,0]))

    nodes_string_ids[d_str_id] = {
        'type' : 'destination_node',
        'features' : { 'n_200' : n200 },
        'int_id' : d_int_id
    }

    list_of_nodes_features.append(torch.tensor([0,0,0]))


# Get les infos des trips
for i, trip in enumerate(trips_list):

    t = trip.split(';')
    start_time = int(t[2])
    end_time = int(t[4])

    # t[2] : start
    # t[4] : end
    # 

    t_str_id = 'n_T{}'.format(i)
    t_int_id = len(nodes_int_ids)
    nodes_int_ids.append(t_str_id)

    nodes_string_ids[t_str_id] = {
        'type' : 'trip',
        'features' : {
            'start_time' : start_time,
            'end_time' : end_time,
            'duration' : end_time - start_time
        },
        'int_id' : t_int_id
    }

    list_of_nodes_features.append(torch.tensor([start_time,end_time,end_time-start_time]))


# Au lieu de refaire toute la logique pour creer le graphe du MDEVSP, on prend le fichier inputProblem{}

src_ids = []
dst_ids = []


list_of_edges_features = []

input_file = open("Networks/Network{}/inputProblem{}_default.in".format(instance, instance), "r")
for line in input_file:
    if "Arcs" in line:
        # ok we are where we want

        for line in input_file:

            if "}" in line:
                break

            # edge : between two nodes

            t = line.split(' ')

            o_str_id = t[0]
            d_str_id = t[1]

            cost = int(t[2])

            energy = int(t[3].replace('[',''))

            # print('{} -> {} : c:{},e:{}'.format(o_str_id, d_str_id, cost, energy))

            if o_str_id in nodes_string_ids and d_str_id in nodes_string_ids:

                o_int_id = nodes_string_ids[o_str_id]['int_id']
                d_int_id = nodes_string_ids[d_str_id]['int_id']

                src_ids.append(o_int_id)
                dst_ids.append(d_int_id)
                
                list_of_edges_features.append(torch.tensor([cost, energy]))

src_ids = torch.tensor(src_ids)
dst_ids = torch.tensor(dst_ids)

G = dgl.graph((src_ids, dst_ids))

print('Nb Nodes : {}'.format(G.num_nodes()))
print('Nb edges : {}'.format(G.num_edges()))


G.ndata['x'] = torch.stack((list_of_nodes_features))

G.edata['w'] = torch.stack((list_of_edges_features))

print(G.ndata['x'])

print(G.edata['w'])
