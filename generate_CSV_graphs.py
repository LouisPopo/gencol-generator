# This file takes an instance : voyages.txt, recharges.txt, hlp.txt and depots.txt and creates a graph :
# Output : nodes.csv and edges.csv with ids and (basics) features

import pickle
from cmath import cos
import enum
from math import ceil
from instances_params import *
import pandas as pd
from glob import glob
import os

df_all_nodes = pd.DataFrame()
df_all_edges = pd.DataFrame()

# Il faudrait qqpart storer un dict entre id(dgl graph) -> instance correspondante
df_all_graphs = pd.DataFrame(columns=['graph_id', 'feat', 'id', 'nb_trip'])

nb_instances = len(glob('Networks/Network*'))

instances_id_to_info = dict()

for instance_id, instance_folder in enumerate(glob('Networks/Network*')):

    instance_info = instance_folder.split('/')[1].replace('Network', '')

    instances_id_to_info[instance_id] = instance_info

    dual_vars_file_path = '{}/dualVarsFirstLinearRelaxProblem{}_default.out'.format(instance_folder, instance_info)

    if not os.path.exists(dual_vars_file_path):
        continue

            
    network_folder = "Networks/Network{}".format(instance_info)

    trips_file = open(network_folder + "/voyages.txt", "r")
    trips_list = trips_file.readlines()
    nb_trip = len(trips_list)

    depot_file = open(network_folder + '/depots.txt', 'r')
    depots_list = depot_file.readlines()
    nb_depots = len(depots_list)

    chrg_stations_file = open(network_folder + '/recharge.txt', 'r')
    chrg_stations_list = chrg_stations_file.readlines()
    nb_charg_stations = len(chrg_stations_list)

    # Get le nb de periodes necessaires
    t_min = 1400
    t_max = 0

    # tripID,startNode,startTime,endNode,endTime,lineNb
    for trip in trips_list:
        trip_end_time = int(trip.split(';')[4])

        if trip_end_time < t_min:
            t_min = trip_end_time
        
        if trip_end_time > t_max:
            t_max = trip_end_time

    # t_min est le temps le plus tot ou un trajet termine
    # t_max est le temps le plus tard qu'un trajet termine

    t_min = (t_min//recharge)*recharge    
    t_max = int(ceil(t_max/recharge)+4)*recharge
        
    periods = [i for i in range(t_min, t_max+recharge, recharge)]
    nb_periods = len(periods)

    # Get les distances hlp

    hlp_file = open(network_folder + '/hlp.txt', 'r')
    hlp = {}
    for line in hlp_file.readlines():
        hlp_info = line.split(';')
        o = hlp_info[0] # Origin
        d = hlp_info[1] # Destination
        t = hlp_info[2] # Time

        if o in hlp:
            hlp[o][d] = int(t)
        else:
            hlp[o] = {d : int(t)}

    def c_trip(n1, n2, e1, s2): # Cost of the hlp trip from node 1 to node 2 between 
        # a trip finishing at n1 at time e1 and starting at n2 at time s2
        travel_time = hlp[n1][n2]
        wait_time = s2 - e1 - travel_time
        cost = int(cost_t*travel_time + cost_w*wait_time)
        total_time = s2 - e1
        return cost, travel_time, wait_time, total_time

    def e_trip(n1, n2, t1, t2): # Energy consumption on the trip from n1 to n2 + waiting time      
        travel_time = hlp[n1][n2]
        dist = speed*travel_time
        waiting_time = t2 - (t1 + travel_time)
        return int(enrgy_km*dist + enrgy_w*waiting_time)

    def e_hlp(n1, n2): # Energy consumption on empty travel between n1 and n2
        travel_time = hlp[n1][n2]
        dist = travel_time*speed
        return int(enrgy_km*dist)

    def c_depot(n1, n2):
        travel_time = hlp[n1][n2]
        cost = int(cost_t*travel_time)
        return cost, travel_time, 0

    def compatible(n1, n2, e1, s2):
        travel_time = hlp[n1][n2]
        return ((e1 + travel_time) <= s2) and ((s2 - e1) <= delta)


    # On creer les noeuds : un par voyage, + recharges
    # On creer les arcs : un entre chaque trips qui peuvent se preceder, + recharges

    # NODES : 

    df_nodes = pd.DataFrame(columns=['name','t_s','t_e','n_s','n_e']) #name, time_start, time_end, node_start, node_end

    for i, depot in enumerate(depots_list):
        d = depot.split(';')
        df_nodes.loc[len(df_nodes)] = ["o_D{}".format(i), 0, 0, d[0], d[0]]
        df_nodes.loc[len(df_nodes)] = ["k_D{}".format(i), t_max, t_max, d[0], d[0]]
    for i, trip in enumerate(trips_list):
        t = trip.split(';')
        df_nodes.loc[len(df_nodes)] = ["n_T{}".format(i), t[2], t[4], t[1], t[3]]
    for i, borne in enumerate(chrg_stations_list):
        b = borne.split(';')
        for t in range(nb_periods):
            df_nodes.loc[len(df_nodes)] = ["w_H{}_P{}".format(i, t), periods[t], periods[t], b[0], b[0]]
            df_nodes.loc[len(df_nodes)] = ["c_H{}_P{}".format(i, t), periods[t], periods[t], b[0], b[0]]

    # 1 par depot (besoin ou non ?)
    # 1 par trip
    # 1 par charging station par time period

    # EDGES : 

    list_edges = []

    for i, trip1 in enumerate(trips_list):

        t1_id,t1_n_s,t1_t_s,t1_n_e,t1_t_e,t1_line,_ = trip1.split(';')   # B3;n3102204;264;n3300440;295;411;
                                # id, node_s ,t_s, node_e ,t_e, line, ''
        
        t1_energy = e_trip(t1_n_s, t1_n_e, int(t1_t_s), int(t1_t_e))
        # depot -> start trip
        # trip + end_trip -> depot
        for k, depot in enumerate(depots_list):
            # depot -> start trip
            depot_node,n200,_ = depot.split(';')
            energy = int(e_hlp(depot_node, t1_n_s))
            cost, travel_time, WWW = c_depot(depot_node, t1_n_s)

            list_edges.append(["o_D{}".format(k), "n_T{}".format(i), cost, energy, travel_time, WWW, travel_time, 0, t1_line])

            energy = int(t1_energy + e_hlp(t1_n_e, depot_node))
            cost, travel_time, WWW = c_depot(t1_n_e, depot_node)
            list_edges.append(["n_T{}".format(i), "k_D{}".format(k), cost, energy, travel_time, WWW, travel_time, t1_line, 0])
            
        # trip1 + end_trip1 -> start_trip2
        for j, trip2 in enumerate(trips_list):
            t2_id,t2_n_s,t2_t_s,t2_n_e,t2_t_e,t2_line,_ = trip2.split(';')
            if compatible(t1_n_e, t2_n_s, int(t1_t_e), int(t2_t_s)):
                energy = int(t1_energy + e_hlp(t1_n_e, t2_n_s))
                cost, travel_time, wait_time, total_time = c_trip(t1_n_e, t2_n_s, int(t1_t_e), int(t2_t_s))
                list_edges.append(["n_T{}".format(i), "n_T{}".format(j), cost, energy, travel_time, wait_time, total_time, t1_line, t2_line])

        
        for h, charging_station in enumerate(chrg_stations_list):
            # trip1 + end_trip1 -> waiting borne
            st_n,nb_b,_ = charging_station.split(';')
            travel_time = hlp[t1_n_e][st_n]
            p = min([t for t in range(nb_periods) if periods[t] >= int(t1_t_e) + travel_time] + [nb_periods+1])
            if p != nb_periods + 1:
                energy = t1_energy + e_hlp(t1_n_e, st_n)
                cost, travel_time, wait_time = c_depot(t1_n_e, st_n)
                list_edges.append(["n_T{}".format(i), "w_H{}_P{}".format(h,p), cost, energy, travel_time, wait_time, periods[p]-int(t1_t_e), t1_line, 0])

            travel_time = hlp[st_n][t1_n_s]
            p = max([t for t in range(nb_periods) if periods[t] <= int(t1_t_s)-travel_time]+[-1])
            if p != -1:
                energy = int(e_hlp(st_n, t1_n_s))
                cost, travel_time, wait_time = c_depot(st_n,t1_n_s)
                list_edges.append(["w_H{}_P{}".format(h,p), "n_T{}".format(i), cost, energy, travel_time, wait_time, int(t1_t_s)-periods[p], 0, t1_line]) # JULIETTE A : t2_line

    for h, charging_station in enumerate(chrg_stations_list):
        st_n,nb_b,_ = charging_station.split(';')
        for p in range(nb_periods - 1):
            # wait -> wait
            list_edges.append(["w_H{}_P{}".format(h,p), "w_H{}_P{}".format(h,p+1), 0, 0, 0, 0, recharge, 0, 0])
            # wait -> charge
            list_edges.append(["w_H{}_P{}".format(h,p), "c_H{}_P{}".format(h,p+1), 0, 0, 0, 0, recharge, 0, 0])
            
            if p >= 1:
                # charge -> charge
                list_edges.append(["c_H{}_P{}".format(h,p), "c_H{}_P{}".format(h,p+1), 0, 0, 0, 0, recharge, 0, 0])
                # charge -> wait    
                list_edges.append(["c_H{}_P{}".format(h,p), "w_H{}_P{}".format(h,p), 0, 0, 0, 0, 0, 0, 0])  

        list_edges.append(["c_H{}_P{}".format(h,p-1), "w_H{}_P{}".format(h,p-1), 0, 0, 0, 0, 0, 0, 0])    

        for k, depot in enumerate(depots_list):
            depot_node,n200,_ = depot.split(';')
            energy = int(e_hlp(st_n, depot_node))
            cost, travel_time, waiting_time = c_depot(st_n,depot_node)
            list_edges.append(["w_H{}_P{}".format(h, nb_periods-1), "k_D{}".format(k), cost, energy, travel_time, wait_time, travel_time, 0, 0])
                                                            #cost, energy, travel time, waiting time, total time, 
    df_edges = pd.DataFrame(list_edges, columns=['src','dst','cost','energy','travel_time','waiting_time','delta_time','line_i','line_j'])

    # One hot encoding of the type of node
    df_nodes['type'] = df_nodes['name'].apply(lambda x : x.split('_')[0])
    for t in ['o', 'k', 'n', 'w', 'c', 'd']:
        df_nodes[t] = (df_nodes['type']==t).astype(int)

    # dictionnaire entre le nom du noeud et un id unique
    name_idxn = dict(df_nodes.reset_index().set_index('name')['index'])

    # Mets les IDS unique des noeuds sur les edges
    df_edges['src_id'] = df_edges['src'].apply(lambda x : name_idxn[x])
    df_edges['dst_id'] = df_edges['dst'].apply(lambda x : name_idxn[x])
    df_edges = df_edges.astype({'cost' : float, 'energy' : float})

    # Relative cost 1 = max, 0.5 = half max cost
    df_edges['c_stand'] = df_edges['cost']/df_edges['cost'].max()
    df_edges['energy_stand'] = df_edges['energy']/sigma_max # A VERIFIER L'UTILITE

    # ON DEVRAIT ALLER CHERCHER ICI LES "REPONSES ET LES AJOUTER AUX NOEUDS"

    # BOUT DE CODE PRIS DE JULIETTE : ON SIMPLIFIE NOTRE GRAPHE PHYSIQUE, ET ON MERGE LES EMPLACEMENTS
    # QUI ONT UNE DISTANCE DE 0 EN UN NOUVEL ENDROIT

    # Dict entre des noeuds dont la distance est 0
    hlp = pd.read_csv(network_folder + '/hlp.txt',sep=';',header=None, usecols=[0,1,2], names=['n1','n2','t'])
    dict_nodes = dict()
    for x in hlp[hlp.t==0].itertuples():
        if x.n1 not in dict_nodes.keys():
            dict_nodes[x.n1] = [x.n2]
        else :
            if x.n2 not in dict_nodes[x.n1]:
                dict_nodes[x.n1].append(x.n2)

    # print(dict_nodes)

    dict_nodes_2 = dict()
    list_node = []
    for node in dict_nodes.keys():
        if node not in list_node:
            for n2 in dict_nodes[node]:
                dict_nodes_2[n2] = node
            list_node.append(node)
            list_node += dict_nodes[node]
    for node in hlp['n1'].unique():
        if node not in dict_nodes_2.keys():
            dict_nodes_2[node] = node
    # print("----")
    # print(dict_nodes_2)

    df_nodes['n_s'] = df_nodes['n_s'].apply(lambda x:dict_nodes_2[x])
    df_nodes['n_e'] = df_nodes['n_e'].apply(lambda x:dict_nodes_2[x])
    for noeud in df_nodes['n_s'].unique():
        df_nodes['s_'+noeud] = (df_nodes['n_s']==noeud).astype(int)
    for noeud in df_nodes['n_e'].unique():
        df_nodes['e_'+noeud] = (df_nodes['n_e']==noeud).astype(int)


    df_trip = pd.read_csv(network_folder + '/voyages.txt',sep=';',header=None,usecols=[0,1,2,3,4,5],
        names=['trip','n_s','t_s','n_e','t_e','line'])
    df_trip['n_s'] = df_trip['n_s'].apply(lambda x:dict_nodes_2[x])
    df_trip['n_e'] = df_trip['n_e'].apply(lambda x:dict_nodes_2[x])

    df_nodes = df_nodes.astype({'t_s': float, 't_e': float})

    df_nodes['nb_dep_10'] = df_nodes.apply(lambda x:len(df_trip.loc[(df_trip.n_s==x.n_s)&(df_trip.t_s>=x.t_s)&
        (df_trip.t_s<=x.t_s+10)])-1 if x.n==1 else 0, axis=1)
    df_nodes['nb_dep_id_10'] = df_nodes.apply(lambda x:len(df_trip.loc[(df_trip.n_s==x.n_s)&(df_trip.n_e==x.n_e)&
        (df_trip.t_s>=x.t_s)&(df_trip.t_s<=x.t_s+10)])-1 if x.n==1 else 0, axis=1)
    df_nodes['nb_fin_10'] = df_nodes.apply(lambda x:len(df_trip.loc[(df_trip.n_e==x.n_e)&(df_trip.t_e<=x.t_e)&
        (df_trip.t_e>=x.t_e-10)])-1 if x.n==1 else 0, axis=1)
    df_nodes['nb_fin_id_10'] = df_nodes.apply(lambda x:len(df_trip.loc[(df_trip.n_e==x.n_e)&(df_trip.t_e<=x.t_e)&
        (df_trip.n_s==x.n_s)&(df_trip.t_e>=x.t_e-10)])-1 if x.n==1 else 0, axis=1)

    df_edges = pd.merge(df_edges, df_nodes[['name','t_e', 'n_e',]], left_on = ['src'], right_on = ['name'])
    df_edges = pd.merge(df_edges, df_nodes[['name','t_s', 'n_s']], left_on = ['dst'], right_on = ['name'])


    # gg = df_edges.groupby(['src', 'n_s']).rank("dense", ascending=True)
    # for key, item in gg:
    #     print(gg.get_group(key), "\n\n")

    df_edges['rg_id'] = df_edges.groupby(['src','n_s'])["t_s"].rank("dense", ascending=True)
    df_edges['rg'] = df_edges.groupby(['src'])["t_s"].rank("dense", ascending=True)

    # df_edges.rename(columns= {'c_x': 'c', 'c_y': 'c_x', 'c':'c_y'}, inplace = True)
    del df_edges['name_x']
    del df_edges['name_y']




    # Get les variables duales : 
    dual_vars = pd.read_csv(network_folder + "/dualVarsFirstLinearRelaxProblem{}_default.out".format(instance_info), sep=' ', names=['name', 'pi_value'])
    dual_vars = dual_vars.astype({'pi_value' : float})

    dual_vars['name'] = dual_vars['name'].apply(lambda x : x.replace('Cover', 'n'))

    df_nodes = pd.merge(df_nodes, dual_vars, how="left", on=['name']).fillna(0) # Fill NA pour les dual vars qui ont pas de match (donc les noeuds pas voyage)

    df_nodes['class'] = df_nodes['pi_value'].apply(lambda x : int(x > 0))

    df_nodes['node_id'] = df_nodes.index

    df_nodes['graph_id'] = instance_id
    df_edges['graph_id'] = instance_id

    df_all_nodes = pd.concat([df_all_nodes, df_nodes], ignore_index=True, axis=0)
    df_all_edges = pd.concat([df_all_edges, df_edges], ignore_index=True, axis=0)

    # Dans le graph, son label est le instance info : comme ca on peut retracer l'info de l'instance associée
    df_all_graphs.loc[len(df_all_graphs)] = [instance_id, 0, instance_id, nb_trip]

    # on devrait avoir un lien entre graph_id et le vrai nom du graph. 

    #df_nodes.to_csv('Networks/Network{}/graph_nodes.csv'.format(instance_info), sep=';')
    #df_edges.to_csv('Networks/Network{}/graph_edges.csv'.format(instance_info), sep=';')

    print('{}/{} done'.format(instance_id + 1, nb_instances))



with open('Networks/instances_id_to_info.pkl', 'wb') as f:
    pickle.dump(instances_id_to_info, f)
    #f.write(str(instances_id_to_info))

# xs_graphs_ids = list(df_all_graphs.loc[df_all_graphs['nb_trip'] <= 600, 'graph_id'])
# s_graphs_ids = list(df_all_graphs.loc[(df_all_graphs['nb_trip'] > 600) & (df_all_graphs['nb_trip'] <= 800), 'graph_id'])
# m_graphs_ids = list(df_all_graphs.loc[(df_all_graphs['nb_trip'] > 800) & (df_all_graphs['nb_trip'] <= 1000), 'graph_id'])
# l_graphs_ids = list(df_all_graphs.loc[df_all_graphs['nb_trip'] > 1000, 'graph_id'])


# if STOP == 0:

df_all_graphs.drop(['nb_trip'], axis=1, inplace=True)

df_all_graphs.to_csv('MDEVSP_dataset/graphs.csv', sep=',', index=False)
df_all_nodes.to_csv('MDEVSP_dataset/nodes.csv', sep=',', index=False)
df_all_edges.to_csv('MDEVSP_dataset/edges.csv', sep=',', index=False)
