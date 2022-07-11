from asyncio import tasks
from datetime import datetime
import enum
import grp
from math import ceil
import os
import random

import numpy as np

random.seed(datetime.now())


fixed_cost = 1000       # Cout d'un vehicule
nb_veh = 20             # Nb. de vehicules disponibles a un depot

sigma_max = 363000      # En Wh
speed = 18/60           # en km/min (en moyenne, sans compter les arrets)
enrgy_km = 1050         # Energie consommee en Wh par km parcouru
enrgy_w = 11000/60      # Energie consommee en Wh par minute passee a l'arret

cost_w = 2              # Cout d'attente par min.
cost_t = 4              # Cout voyage a vide par min.
delta = 45              # Temps. max entre la fin d'un trajet et le debut d'un autre

p = 15                  # Nb. de periodes d'echantillonage pour la recharge


def create_gencol_file(
    list_pb, 
    fixed_cost=1000, 
    nb_veh=20, 
    sigma_max=363000, 
    speed=18/60, 
    enrgy_km=1050, 
    enrgy_w=11000/60, 
    cost_w=2, 
    cost_t=4, 
    delta=45, 
    p=15, 
    recharge=15, 

    ## 
    path_to_networks = 'Networks', 
    dual_variables_file_name='', 
    percentage_ineq = 0,    # Le pourcentage d'inégalités qu'on aurait "trouvé", donc qu'on est un peu confiant, mais on 
                            # on peut quand même faire des erreurs
    
    #nb_grps = 0, 
    #take_absolute_value = False, 
    
    #percentage_wrong = 0,
    add_pairwise_inequalities = True 
    #use_strong_task = True, 
    #test_new_grp = False, 
    #new_grp_val_range = 0
    ##

    ):

    for pb in list_pb:

        network_folder = '{}/Network{}'.format(path_to_networks, pb)

        trips_file = open(network_folder + "/voyages.txt", "r")
        trips_list = trips_file.readlines()
        nb_trip = len(trips_list)

        depot_file = open(network_folder + '/depots.txt', 'r')
        depots_list = depot_file.readlines()
        nb_depots = len(depots_list)

        chrg_stations_file = open(network_folder + '/recharge.txt', 'r')
        chrg_stations_list = chrg_stations_file.readlines()
        nb_charg_stations = len(chrg_stations_list)

        # Read dual variables values
        dual_variables = []
        if dual_variables_file_name != '':
            dual_variables_file = open(network_folder + '/' + dual_variables_file_name, 'r')
            dual_variables_list = dual_variables_file.read().splitlines()
            for line in dual_variables_list:

                dual_variable, value = line.split(' ')
                value = float(value)

                # Pour l'instant on laisse faire les +1000 (cout fixe) et les <0
                # Finalement on garde le < 0
                if value < 850 and "Max" not in dual_variable and "Count" not in dual_variable:
                    dual_variables.append((dual_variable, value))

        dual_variables.sort(key = lambda pair: pair[1], reverse=True)
        nb_dual_variables = len(dual_variables)

        # On créé les inégalités duales : 

        # 1) On évalue chaque paire de variables duales, on définit une relation >=, on trouve une ou plusieurs séries (assez longue)
        # et ça donne notre ou nos séries d'inégalités

        # 2) On groupes les variables duales (en range de 0 à 4, 4 à 8, etc.) et de la on est capable d'établir des inégalités ENTRE
        # groupes. 

        tasks_in_new_inequalities = set()
        inequalities = []

        nb_inequalities = 0

        if dual_variables_file_name != '':
            
            nb_inequalities = int(percentage_ineq * nb_dual_variables)

            if add_pairwise_inequalities:

                s = random.sample(dual_variables, nb_inequalities)
                # sort du plus grand au plus petit
                s.sort(key=lambda pair: pair[1], reverse=True)

                for i in range(len(s) - 1):

                    for j in range(i + 1, len(s)):

                        # unique paire : s[i] - s[j]

                        diff = abs(s[i][1] - s[j][1])

                        # on sait que s[i] >= s[j], mais en fonction de la différence, on calcul une probabilité de se tromper

                        v = random.random() # return a value between [0,1) 1 never there
                        
                        treshold = 1 # en fonction de la diff

                        if v <= treshold:
                            # ok
                            pi_1 = s[i][0]
                            pi_2 = s[j][0]

                        else:
                            # mauvaise inegalite
                            pi_1 = s[j][0]
                            pi_2 = s[i][0]    

                        tasks_in_new_inequalities.add(pi_1)
                        tasks_in_new_inequalities.add(pi_2)
                        
                        # techniquement non, on ajoute pas la, on note les relations
                        inequalities.append((pi_1, pi_2))

                        # compute odds of having wrong inequality

            else:

                # on groupe les variables duales
                s = random.sample(dual_variables, nb_inequalities)

                # sort du plus grand au plus petit
                s.sort(key = lambda pair: pair[1], reverse=True)

                ineq_groups = []

                max_val = int(s[0][1])
                min_val = int(s[-1][1])

                group_range = 6 # VARIABLE

                for i in range(max_val, min_val, -group_range):
                    
                    group = [x for x in s if (i < x[1]) and (x[1] <= i - group_range)]

                    ineq_groups.append(group)


                    # on les groupe, mais certaines variables duales devraient êtres placés de façon aléatoire (plus de chances d'être
                    # dans un groupe proche)
                
                # Groupes sont en ordre du plus grand au plus petit
                ineq_groups = [g for g in ineq_groups if g != []]

                for g, group in enumerate(ineq_groups):

                    for i in range(len(group) - 1) : 
                        
                        # Ici, c'est les inégalités "inter groupes"

                        pi_1 = group[i][0]
                        pi_2 = group[i+1][0]

                        tasks_in_new_inequalities.add(pi_1)
                        tasks_in_new_inequalities.add(pi_2)
                        inequalities.append((pi_1, pi_2))

                    # pour les n-1 groupes (sauf le dernier)
                    if g < len(ineq_groups) - 1:

                        # on rajoute l'inégalité entre-groupes

                        # le dernier du drenier groupe
                        pi_1 = group[-1][0]
                        pi_2 = ineq_groups[g + 1][0]
                        # le premier du prochain groupe

                        inequalities.append((pi_1, pi_2))
            
            

        output_file_path = "gencol_files/" + pb
        if not os.path.exists(output_file_path):
            os.mkdir(output_file_path)

        output_file_name = "inputProblem" + pb 
        
        # if percentage_wrong > 0:
        #     output_file_name += "_{}_{}_W_{}".format(int(nb_grps), grp_size, nb_wrong)
        # else:
        #     output_file_name += '_P_{}_{}'.format(int(nb_grps), grp_size)
        
        if add_pairwise_inequalities:
            output_file_name += "_P_{}".format(len(inequalities))
        else:
            output_file_name += "_G_{}_{}".format(len(inequalities, group_range))

        #output_file_name += "_default"

        print(output_file_name)
            
        output_file = open(output_file_path + "/" + output_file_name + ".in", "w")
        
    

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
            if row_name not in tasks_in_new_inequalities:
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
            if row_name not in tasks_in_new_inequalities:
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
        
        
        for i, ineq in enumerate(inequalities):
            pi_1 = ineq[0]
            pi_2 = ineq[1]
            # pi_1 >= pi_2
            cols_string += "Y_" + str(i) + " 0 (" + pi_1 + " -1) (" + pi_2 + " 1);\n"
        
        
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

        


