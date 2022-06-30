from asyncio import tasks
from datetime import datetime
import enum
import grp
from math import ceil
import os
import random
from re import S

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


def create_gencol_file(list_pb, fixed_cost=1000, nb_veh=20, sigma_max=363000, speed=18/60, enrgy_km=1050, enrgy_w=11000/60, cost_w=2, cost_t=4, delta=45, p=15, recharge=15, path_to_networks = 'Networks', dual_variables_file_name='', percentage_ineq = 0, nb_grps = 0, take_absolute_value = False, percentage_wrong = 0, use_strong_task = True, test_new_grp = False, new_grp_val_range = 0):

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
                if value < 850 and not take_absolute_value and "Max" not in dual_variable and "Count" not in dual_variable:
                    dual_variables.append((dual_variable, value))
                elif value < 850 and take_absolute_value and "Max" not in dual_variable and "Count" not in dual_variable:
                    dual_variables.append((dual_variable, abs(value)))

        dual_variables.sort(key = lambda pair: pair[1], reverse=True)
        nb_dual_variables = len(dual_variables)

        # On créé les inégalités
        # 
        # Random : on créer X inégalités en sélectionnant deux variables duales aléatoires
        # Order : on selection X+1 variables duales en ordre de façon aléatoire et on créé X inégalités entre elles.

        # Random :
        tasks_in_new_inequalities = set()
        inequalities = []

        nb_inequalities = 0
        grp_size = 0

        if dual_variables_file_name != '':
            
            nb_inequalities = int(percentage_ineq * nb_dual_variables)

            grp_size = int(nb_inequalities/nb_grps)

            if percentage_wrong == 0:

                nb_wrong = 0

                if test_new_grp:

                    s = random.sample(dual_variables, grp_size)
                    # On sort du plus petit au plus grand!
                    s.sort(key = lambda pair: pair[1])

                    min_val = s[0][1]
                    max_val = s[-1][1]

                    ineq_groups = []

                    current_min_val = min_val

                    current_group = []

                    for i,d in enumerate(s):

                        if d[1] <= current_min_val + new_grp_val_range:
                            current_group.append(d)
                        else:
                            ineq_groups.append(current_group)
                            current_group = []

                            current_min_val = d[1]
                            
                            #while d[1] > current_min_val + new_grp_val_range:
                            #    current_min_val += new_grp_val_range
                            
                            current_group.append(d)

                    # On reverse la liste pour avoir les plus grosses valeurs en debut de liste
                    ineq_groups.reverse()

                    line = ''

                    for g in ineq_groups:

                        min_val_g = min(g, key = lambda x : x[1])
                        max_val_g = max(g, key = lambda x : x[1])

                        #for v in g:

                            # if (v[1] >= 0 and v[1] < 1) or (v[1] <= 0 and v[1] > -1):
                            #    print(v)

                        #line += '[ {} - ( {} ) - {} ] '.format(max_val_g[1], len(g), min_val_g[1])
                    
                    #print(line)
                        



                    # On test des inge du genre : 
                    # [pi1, pi2, pi3], [pi4, pi5], [pi6]
                    # pi1 >= pi4 >= pi6
                    # pi2 >= pi5

                    #print(", ".join([str(len(x)) for x in ineq_groups]))

                    min_nb_dual_var = min(len(x) for x in ineq_groups)
                    max_nb_dual_var = max(len(x) for x in ineq_groups)

                    nb_groups = len(ineq_groups)

                    nb_series = 0

                    for _ in range(max_nb_dual_var):
                        # on va chercher une variable duale par groupe non vide, ca nous fait notre liste
                        
                        serie = []
                        
                        for i, g in enumerate(ineq_groups):

                            if len(g) > 0:

                                nb_values_take = min(len(g), 9)

                                #dual_var = random.sample(g, 1)[0]
                                # On prend 5 valeurs par groupes 
                                dual_vars = random.sample(g, nb_values_take)
                                # On les classe, 
                                #dual_vars.sort(key=lambda x : x[1], reverse=True)
                                # Et on rajoute une inegalite entre eux aussi
                                for d in dual_vars:

                                    serie.append(d)

                                    ineq_groups[i].remove(d)


                        line = ''
                        for s in serie :
                            line += '{} '.format(s[1])
                        print(line)

                        if len(serie) > int(0.5*nb_groups):

                            nb_series += 1

                            for d in range(len(serie) - 1):

                                pi_1 = serie[d][0]
                                pi_2 = serie[d + 1][0]

                                tasks_in_new_inequalities.add(pi_1)
                                tasks_in_new_inequalities.add(pi_2)

                                inequalities.append((pi_1, pi_2))

                            line = ''
                            for e in serie:
                                line += '{} >= '.format(e[1])
                            #print(line)
                        
                        else:
                            break



                    # for g1 in range(len(ineq_groups) - 1):
                        

                    #     for g1_dual_var in ineq_groups[g1]:
                            
                    #         pi_1 = g1_dual_var[0]
                    #         tasks_in_new_inequalities.add(pi_1)

                    #         for g2_dual_var in ineq_groups[g1 + 1]:

                    #             pi_2 = g2_dual_var[0]
                    #             tasks_in_new_inequalities.add(pi_2)

                    #             inequalities.append((pi_1, pi_2))

                        # On prend une var duale du g1+1, et chaque var duale du g1 >= celle random
                        # g2_dual_var = random.sample(ineq_groups[g1+1], 1)[0]
                        # pi_2 = g2_dual_var[0]
                        # tasks_in_new_inequalities.add(pi_2)

                        # for g1_dual_var in ineq_groups[g1]:
                        #     pi_1 = g1_dual_var[0]

                        #     tasks_in_new_inequalities.add(pi_1)
                        #     inequalities.append((pi_1, pi_2))

                    print('{} ineq in total'.format(len(inequalities)))
                    
                    
                            

                else:

                    for g in range(nb_grps):

                        s = random.sample(dual_variables, grp_size)

                        s.sort(key = lambda pair: pair[1], reverse=True)

                        for d in s : dual_variables.remove(d)

                        for i in range(grp_size - 1):
                            pi_1 = s[i][0]
                            pi_2 = s[i+1][0]

                            tasks_in_new_inequalities.add(pi_1)
                            tasks_in_new_inequalities.add(pi_2)
                            inequalities.append((pi_1, pi_2))
            
            else:

                # On ajoute des inégalités "fausses"

                for g in range(nb_grps):
                    
                    # SEQUENTIAL INEQUIALITIES

                    s = random.sample(dual_variables, grp_size)

                    nb_wrong = int(percentage_wrong * grp_size) # le tiers est mauvais

                    print("number of wrong ineq : {}".format(nb_wrong))

                    for _ in range(nb_wrong):

                        i = random.randrange(0, len(s) - 1)
                        
                        old_value = s[i][1]
                        new_value = old_value

                        while old_value == new_value:
                            new_value = random.randrange(0,55)

                        s[i] = (s[i][0], new_value)
                        

                    s.sort(key = lambda pair: pair[1], reverse=True)

                    # #for d in s : dual_variables.remove(d)

                    for i in range(grp_size - 1):
                        pi_1 = s[i][0]
                        pi_2 = s[i+1][0]

                        tasks_in_new_inequalities.add(pi_1)
                        tasks_in_new_inequalities.add(pi_2)
                        inequalities.append((pi_1, pi_2))

                    # PAIRWISE INEQUALITIES

                    # i = 0
                    # while i < grp_size:

                    #     r = random.sample(dual_variables, 2)

                    #     if r[0][1] >= r[1][1]:
                    #         pi_1 = r[0][0]
                    #         pi_2 = r[1][0]
                    #     else:
                    #         pi_1 = r[1][0]
                    #         pi_2 = r[0][0]

                    #     if (pi_1, pi_2) not in inequalities:
                    #         tasks_in_new_inequalities.add(pi_1)
                    #         tasks_in_new_inequalities.add(pi_2)
                    #         inequalities.append((pi_1, pi_2))
                    #         i += 1

            # Sequence : 
            # else :
            #     sub_list = random.sample(dual_variables, nb_inequalities+1)
            #     sub_list.sort(key = lambda pair: pair[1], reverse=True)

            #     for i in range(nb_inequalities):
            #         pi_1 = sub_list[i][0]
            #         pi_2 = sub_list[i+1][0]

            #         tasks_in_new_inequalities.add(pi_1)
            #         tasks_in_new_inequalities.add(pi_2)
            #         inequalities.append((pi_1, pi_2))


        output_file_path = "gencol_files/" + pb
        if not os.path.exists(output_file_path):
            os.mkdir(output_file_path)

        output_file_name = "inputProblem" + pb 
        
        # if percentage_wrong > 0:
        #     output_file_name += "_{}_{}_W_{}".format(int(nb_grps), grp_size, nb_wrong)
        # else:
        #     output_file_name += '_P_{}_{}'.format(int(nb_grps), grp_size)
        
        if test_new_grp:
            output_file_name += "_{}_{}".format(len(inequalities), nb_series)
        elif nb_inequalities > 0 and grp_size > 0:
            output_file_name += "_{}_{}_W_{}".format(int(nb_grps), grp_size, nb_wrong)
        else:
            output_file_name += "_default"

        print(output_file_name)
            
        # if id != '':
        #     output_file_name += "_" + str(id)
        output_file = open(output_file_path + "/" + output_file_name + ".in", "w")
        # step = 1
        # tasks_in_new_inequalities = set()
        # inequalities = []
        # for i in range(0, nb_inequalities*step, step):
        #     pi_1 = dual_variables[i][0]
        #     pi_2 = dual_variables[i+1][0]
        #     print("{} >= {}".format(pi_1, pi_2))
        #     inequalities.append((pi_1, pi_2))
        #     tasks_in_new_inequalities.add(pi_1)
        #     tasks_in_new_inequalities.add(pi_2)
        # print(tasks_in_new_inequalities)
        
        

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

            if use_strong_task:
                rows_string += row_name + " =1"
                if row_name not in tasks_in_new_inequalities:
                    rows_string += " TaskStrong"
            else:
                rows_string += row_name + " >= 1"
            
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
            if row_name not in tasks_in_new_inequalities and use_strong_task:
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

        


