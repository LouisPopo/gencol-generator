# This file takes an instance : voyages.txt, recharges.txt, hlp.txt and depots.txt and creates a graph :
# Output : nodes.csv and edges.csv with ids and (basics) features

from cmath import cos
import enum
from math import ceil
from instances_params import *
import pandas as pd

instance_name = "4b_4_0"
network_folder = "Networks/Network{}".format(instance_name)

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
    wait_time = s2 - (e1 + travel_time)
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
    return ((e1 + travel_time) <= s2) and (s2 - e1) <= delta


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
            cost, travel_time, wait_time, total_time = c_trip(t1_n_e, t1_n_s, int(t1_t_e), int(t2_t_s))
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
    for p in range(nb_periods):
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

df_edges = pd.DataFrame(list_edges, columns=['src','dst','c','r1','t_d','t_w','delta','line_i','line_j'])


df_nodes.to_csv('DGL_graph/nodes{}.csv'.format(instance_name), sep='\t')
df_edges.to_csv('DGL_graph/edges{}.csv'.format(instance_name), sep='\t')