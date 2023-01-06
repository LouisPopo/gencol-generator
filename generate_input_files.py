# Apres avoir genere un certain nombre dinstances, on utilise ce script pour generer les fichiers dentrees pour gencol
# C'est un script different car on peut vouloir generer des fichiers inputs differents mais garder les memes instances en entree deja crees

# Va generer un fichier inputProblem... par instance

# En ce moment, le nombre de vehicules et de bornes est fixe

import os

from instances_params import *
from math import ceil

# Parametre avec un impact : nb_veh

# 1. Va chercher tous les folder dans Networks et genere un fichier inputProblem... par folder (par instance)

for folder in os.listdir('Networks'):

    if '9999' not in folder:
        continue

    #
    instance_seed = int(folder.split('_')[-1])
    if instance_seed < 100:
        continue
    #

    print(folder)

    network_folder = 'Networks/{}'.format(folder)

    instance_name = folder.replace('Network', '')

    trips_file = open(network_folder + "/voyages.txt", "r")
    trips_list = trips_file.readlines()
    nb_trip = len(trips_list)

    depot_file = open(network_folder + '/depots.txt', 'r')
    depots_list = depot_file.readlines()
    nb_depots = len(depots_list)

    chrg_stations_file = open(network_folder + '/recharge.txt', 'r')
    chrg_stations_list = chrg_stations_file.readlines()
    nb_charg_stations = len(chrg_stations_list)

    # sufix : default
    output_file_name = "inputProblem{}_default.in".format(instance_name)       
    
    output_file = open("Networks/{}/{}".format(folder, output_file_name), "w")
    

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
        return int(cost_t*travel_time + cost_w*wait_time)

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
        return int(cost_t*travel_time)

    def compatible(n1, n2, e1, s2):
        travel_time = hlp[n1][n2]
        return ((e1 + travel_time) <= s2) and (s2 - e1) <= delta


    # Write on output file
    
    # ROWS
    rows_string = "Rows = {\n"

    for trip in range(nb_trip):
        row_name = "Cover_T" + str(trip)

        rows_string += row_name + " =1"
        
        rows_string += " TaskStrong"
        
        rows_string += ";\n"

    for depot in range(nb_depots):
        rows_string += "Count_D" + str(depot) + " = 0;\n"
    for c, chrg_station in enumerate(chrg_stations_list):
        nb_chrg_point = chrg_station.split(";")[1]
        for t in range(nb_periods):
            rows_string += "Max_H" + str(c) + "_P" + str(t) + " <= " + str(nb_chrg_point) + ";\n"
    rows_string += "};\n\n" 

    output_file.write(rows_string)

    # TASKS

    tasks_string = "Tasks = {\n"
    for trip in range(nb_trip):
        
        row_name = "Cover_T" + str(trip)

        tasks_string += "t_T" + str(trip) + " " + row_name
        
        # TEST NO STRONG
        tasks_string += " Strong"
        
        tasks_string += ";\n"

    for depot in range(nb_depots):
        tasks_string += "t_D" + str(depot) + "o Weak;\n"
        tasks_string += "t_D" + str(depot) + "k Weak;\n"
    tasks_string += "};\n\n"

    output_file.write(tasks_string)

    # COLUMNS

    cols_string = "Columns = {\n"
    for depot in range(nb_depots):
        cols_string += "Veh_D" + str(depot) + " " + str(fixed_cost) + " Int [0 " + str(nb_veh) + "] (Count_D" + str(depot) + " -1);\n"
    
    cols_string += "};\n\n"

    output_file.write(cols_string)

    # RESSOURCES
    
    output_file.write("Resources = {\nr_SoC_Inv;\nr_Rch;\nr_Not_Rch;\n};\n\n")

    # NODES

    nodes_string = "Nodes = {\n"
    for depot in range(nb_depots):
        nodes_string += "o_D" + str(depot) + " [0 0] [0 0] [0 0] " + "t_D" + str(depot) + "o (Count_D" + str(depot) + " 1);\n"
        nodes_string += "k_D" + str(depot) + " [0 " + str(sigma_max) +"] [0 1] [0 1] " + "t_D" + str(depot) + "k;\n"
    for trip in range(nb_trip):
        nodes_string += "n_T" + str(trip) + " [0 " + str(sigma_max) +"] [0 0] [0 0] " + "t_T" + str(trip) + ";\n" 
    for chrg_station in range(nb_charg_stations):
        for t in range(nb_periods):
            nodes_string += "w_H" + str(chrg_station) + "_P" + str(t) + " [0 " + str(sigma_max) + "] [0 1] [0 1];\n"
            nodes_string += "c_H" + str(chrg_station) + "_P" + str(t) + " [0 " + str(sigma_max) + "] [1 1] [0 1];\n"
    nodes_string += "};\n\n"

    output_file.write(nodes_string) 

    # ARCS

    arcs_string = "Arcs = {\n"
    for i, trip1 in enumerate(trips_list):

        t1_id,t1_node_start,t1_time_start,t1_node_end,t1_time_end,t1_line_nb,_ = trip1.split(';')
        t1_time_start = int(t1_time_start)
        t1_time_end = int(t1_time_end)

        for k, depot in enumerate(depots_list):

            # depot -> start trip
            depot_node,depot_nb,_ = depot.split(';')
            e = e_hlp(depot_node, t1_node_start)
            c = c_depot(depot_node, t1_node_start)
            arcs_string += "o_D" + str(k) + " n_T" + str(i) + " " + str(c) + " [" + str(e) + " 0 0];\n"

            # start trip -> end trip -> depot
            e = e_trip(t1_node_start, t1_node_end, t1_time_start, t1_time_end) + e_hlp(t1_node_end, depot_node)
            c = c_depot(t1_node_end, depot_node)
            arcs_string += "n_T" + str(i) + " k_D" + str(k) + " " + str(c) + " [" + str(e) + " 0 0];\n"

        for j, trip2 in enumerate(trips_list):
            
            t2_id,t2_node_start,t2_time_start,t2_node_end,t2_time_end,t2_line_nb,_ = trip2.split(';')
            t2_time_start = int(t2_time_start)
            t2_time_end = int(t2_time_end)

            if compatible(t1_node_end, t2_node_start, t1_time_end, t2_time_start):
                e = e_trip(t1_node_start, t1_node_end, t1_time_start, t1_time_end) + e_hlp(t1_node_end, t2_node_start)
                c = c_trip(t1_node_end, t2_node_start, t1_time_end, t2_time_start)
                arcs_string += "n_T" + str(i) + " n_T" + str(j) + " " + str(c) + " [" + str(e) + " 0 0];\n"

        for h, chrg_station in enumerate(chrg_stations_list):

            chrg_station_node,_,_ = chrg_station.split(';')
            t = hlp[t1_node_end][chrg_station_node]
            p = min([w for w in range(nb_periods) if periods[w] >= t1_time_end + t] + [nb_periods + 1])

            # C'est quoi ces arcs

            if p != nb_periods + 1:
                e = e_trip(t1_node_start, t1_node_end, t1_time_start, t1_time_end) + e_hlp(t1_node_end, chrg_station_node)
                c = c_depot(t1_node_end, chrg_station_node)
                arcs_string += "n_T" + str(i) + " w_H" + str(h) + "_P" + str(p) + " " + str(c) + " [" + str(e) + " 0 1];\n"

            t = hlp[chrg_station_node][t1_node_start]
            p = max([w for w in range(nb_periods) if periods[w] <= t1_time_start - t] + [-1])

            if p != -1:
                e = e_hlp(chrg_station_node, t1_node_start)
                c = c_depot(chrg_station_node, t1_node_start)
                arcs_string += "w_H" + str(h) + "_P" + str(p) + " n_T" + str(i) + " " + str(c) + " [" + str(e) + " -1 0];\n"

    for h, chrg_station in enumerate(chrg_stations_list):
        chrg_station_node,_,_ = chrg_station.split(';')

        for p in range(nb_periods - 1):

            arcs_string += "w_H" + str(h) + "_P" + str(p) + " w_H" + str(h) + "_P" + str(p+1) + " 0 [0 0 0];\n"

            arcs_string += "w_H" + str(h) + "_P" + str(p) + " c_H" + str(h) + "_P" + str(p+1) + " 0 [-" + str(recharge) + " 1 -1] (Max_H" + str(h) + "_P" + str(p) + " 1);\n"

            if p >= 1:

                arcs_string += "c_H" + str(h) + "_P" + str(p) + " c_H" + str(h) + "_P" + str(p+1) + " 0 [-" + str(recharge) + " 0 0] (Max_H" + str(h) + "_P" + str(p) + " 1);\n"

                arcs_string += "c_H" + str(h) + "_P" + str(p)  + " w_H" + str(h) + "_P" + str(p) + " 0 [0 0 0];\n"

        arcs_string += "c_H" + str(h) + "_P" + str(nb_periods - 1) + " w_H" + str(h) + "_P" + str(nb_periods - 1) + " 0 [0 0 0];\n"

        for k, depot in enumerate(depots_list):
            depot_node,depot_nb,_ = depot.split(';')
            e = e_hlp(chrg_station_node, depot_node)
            c = c_depot(chrg_station_node, depot_node)

            arcs_string += "w_H" + str(h) + "_P" + str(nb_periods - 1) + " k_D" + str(k) + " " + str(c) + " [" + str(e) + " 0 0];\n"

    arcs_string += "};\n\n"

    output_file.write(arcs_string)

    # NETWORKS
    networks_string = "Networks = {\n"
    for depot in range(nb_depots):
        networks_string += "net_D" + str(depot) + " o_D" + str(depot) + " (k_D" + str(depot) + ");\n"
    networks_string += "};"

    output_file.write(networks_string)