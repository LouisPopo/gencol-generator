from datetime import datetime
from itertools import cycle
from math import ceil
from operator import mod

import os
import random
import numpy as np
import networkx as nx
import time
#import matplotlib.pyplot as plt

from networkx import bellman_ford_path, find_cycle, NetworkXNoCycle


NAME = 0
VALUE = 1

random.seed(datetime.now())

class IneqGraph:

    def __init__(self, nb_nodes):

        self.indice_to_node_name = []
        self.node_name_to_indice = {}

        self.graph = nx.DiGraph()

        self.adj_matrix = np.ones((nb_nodes, nb_nodes)) * np.inf
    
    def add_node(self, node_name):

        self.node_name_to_indice[node_name] = len(self.indice_to_node_name)
        self.indice_to_node_name.append(node_name)

        self.graph.add_node(node_name)

    def add_edge(self, from_node_name, to_node_name, value, prob_right):

        # on ajoute u->v seulement s'il n'existe pas un i t.q. u->i->v existe

        # for middle_node in self.graph.nodes:

        #     if self.graph.has_edge(from_node_name, middle_node) and self.graph.has_edge(middle_node, to_node_name):
        #         return

        self.graph.add_weighted_edges_from([(from_node_name, to_node_name, value)], 'weight', prob=prob_right)

        #self.adj_matrix[self.node_name_to_indice[from_node_name]][self.node_name_to_indice[to_node_name]] = value

    def get_indice_from_node_name(self, node_name):
        return self.node_name_to_indice[node_name]

    def get_node_name_from_indice(self, indice):
        return self.indice_to_node_name[indice]

    def remove_edge_libr(self, u , v):
        self.adj_matrix[self.node_name_to_indice[u]][self.node_name_to_indice[v]] = np.inf

    def remove_edge_hand(self, u, v):
        self.graph.remove_edge(u, v)

    def remove_cycles_libr(self):
        

        while (True) :

            try: 
                cycle =  find_cycle(self.graph, source='Source')

                # print('Found cycle')
                # print(cycle)

                for e in cycle:
                    
                    # On devrait choisir le quel on enleve!
                    self.graph.remove_edge(e[0], e[1])
            
            except NetworkXNoCycle:
                break

        
    def bellman_ford_libr(self):

        path = bellman_ford_path(self.graph, source='Source', target='Sink')

        return False, path

    def bellman_ford_hand(self):

        # Algo. implémenté à la main

        # Ford-Bellman algorithm : 
        # Trouve le "shortest path" from source -> sink
        # Si circuit (negatif) est trouve, on retourne True + le circuit
        # Sinon, on retourne False + le plus court chemin entre source -> sink (la plus longue serie)

        # STEP 1 : Initialize distances and parents
        dist = [np.inf] * ( self.graph.number_of_nodes())
        parent = [-1] * ( self.graph.number_of_nodes())
        dist[self.node_name_to_indice['Source']] = 0

        print('Number of nodes : {}'.format(self.graph.number_of_nodes()))
        print('Number of edges : {}'.format(self.graph.number_of_edges()))

        nb_total_iter = 0
        nb_nodes_iter = 0

        #print('Step1')

        iter = 0

        affected_nodes = set(self.graph.nodes)
        edges_to_iter = set()
        #print(affected_nodes)

        # STEP 2 : Commpute shortest distances (?)
        for _ in range(self.graph.number_of_nodes() - 1):

            #print('Nb affected nodes : {}'.format(len(affected_nodes)))

            edges_to_iter.clear()
            for node in affected_nodes:
                #print(node)
                for e in list(self.graph.in_edges(node)):
                    edges_to_iter.add(e)
                for e in list(self.graph.out_edges(node)):
                    edges_to_iter.add(e)
            
            has_change = False

            affected_nodes.clear()

            #print('Number of edges to iter : {}'.format(len(edges_to_iter)))

            for e in edges_to_iter:
            #for e in list(self.graph.edges):
                
                iter += 1

                u = self.node_name_to_indice[e[0]]
                v = self.node_name_to_indice[e[1]]
                w = self.graph.get_edge_data(e[0], e[1])['weight']

                if dist[u] != np.inf and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    parent[v] = u
                    has_change = True
                    affected_nodes.add(self.indice_to_node_name[v])
                    #affected_nodes.append(self.indice_to_node_name[v])

                
            
                # if iter % 1000000 == 0:
                #     print(iter)

            if len(affected_nodes) == 0:
                break
            
            #rint(has_change)
        #print('Step2')
        #print('Nb nodes iters : {}'.format(nb_nodes_iter))
        #print('Nb total iters : {}'.format(nb_total_iter))
        #print('----')

        #print('Finished STEP 2')

        # STEP 3 : Check for negative weight cycle
        C = -1
        for e in list(self.graph.edges):

            u = self.node_name_to_indice[e[0]]
            v = self.node_name_to_indice[e[1]]
            w = self.graph.get_edge_data(e[0], e[1])['weight']

            if dist[u] != np.inf and dist[u] + w < dist[v]:
                C = v
                break

        #print('Finished STEP 3')

        # STEP 4 : Construct the negative cycle or shortest path to return
        if C != -1:

            for _ in range(self.graph.number_of_nodes() - 1):
                C = parent[C]

            cycle = []
            v = C

            while(True):
                cycle.append(self.indice_to_node_name[v])
                if (v == C and len(cycle) > 1):
                    break
                v = parent[v]

            cycle.reverse()

            return True, cycle 
        else:

            v = self.node_name_to_indice['Sink']
            serie = ['Sink']

            while self.indice_to_node_name[v] != 'Source':
                
                v = parent[v]

                if v == -1:
                    # No parent and we didn't reach source. No more paths
                    break

                serie.append(self.indice_to_node_name[v])
            
            serie.reverse()

            return False, serie

    def get_ineq_series_libr(self):

        # Utilise une matrice adjacence, utilise un algo de librairie

        # 1. Retirer tous les circuits de somme negatives
        # 2. Jusqu'a ce qu'il reste des chemins entre source et sink:
        #       2.1. Trouver le chemin le plus court
        #       2.2. Le retirer du graphe
        # 3. Retourner tous les plus courts chemins : ce sont les inegalites a imposer. 

        start_time = time.time()

        ineq_series = []

        print('Before Removing cycles: {}'.format(self.graph.number_of_edges()))

        print(" === ")
        print("Removing cycles")
        self.remove_cycles_libr()

        print('Removed cycles in {} seconds'.format(time.time() - start_time))
        print('After Removing cycles: {}'.format(self.graph.number_of_edges()))
        

        print(" === ")
        
        print('Finding Series')

        while(True):

            #print(' ======= ')
            
            #has_neg, l = self.get_serie_from_scipy_bellman_ford()
            has_neg, path = self.bellman_ford_libr()

            #print('Found {} series of len : {}'.format(len(ineq_series) + 1, len(l)))

            if not has_neg:

                #print('No neg')

                if len(path) <= 3:
                    break

                #print('No negative serie : ')
                #print(l)

                serie = path[1:-1]

                for i in range(len(serie)):

                    u = path[i]
                    v = path[i+1]

                    self.graph.remove_edge(u, v)
                
                ineq_series.append(serie)

                # if len(ineq_series) >= 100:
                #     break


        print(" Found all series in {} seconds".format(time.time() - start_time))          
        return ineq_series

    def get_ineq_series_hand(self):

        # Utilise graph networks, fait l'algo à la main

        # 1. Retirer tous les circuits de somme negatives
        # 2. Jusqu'a ce qu'il reste des chemins entre source et sink:
        #       2.1. Trouver le chemin le plus court
        #       2.2. Le retirer du graphe
        # 3. Retourner tous les plus courts chemins : ce sont les inegalites a imposer. 

        ineq_series = []

        nb_cycles = 0

        first_no_cycle = True

        while(True):

            #print(' ======= ')
            
            #has_neg, l = self.bellman_ford_hand()
            #print(has_path())
            c = find_cycle(G=self.graph, source='Source')
            print(c)
            p = bellman_ford_path(self.graph, 'Source', 'Sink')
            print(p)

            #print('Found serie {} of len {}'.format(len(ineq_series) + 1, len(l)))
            
            # TEST
            #has_neg_l, l_l = self.bellman_ford_libr()
            # TEST

            #print(l)
            #print(l_l)
            # if l != l_l:
            #     print('WOWOOO NOT Both algo same')

            if not has_neg:

                if first_no_cycle:
                    print('After removing cycle : {}'.format(self.graph.number_of_edges()))
                    first_no_cycle = False



                if len(l) <= 3:
                    print('break')
                    break

                print('Found serie {} of len {}'.format(len(ineq_series) + 1, len(l)))

                #print('No negative serie : ')
                #print(l)

                ineq_series.append(l)

                for i in range(len(l) - 1):

                    u = l[i]
                    v = l[i+1]

                    if u == 'Source' or v == 'Sink':
                        continue
                    
                    #print("Removing {} -> {}".format(u, v))
                    self.remove_edge_hand(u, v)

                    # TEST
                    #self.remove_edge_libr(u, v)
                    # TEST

                if len(ineq_series) > 100:
                    break

            else:

                #print('Got cycle')

                nb_cycles += 1
            
                #print('Negative serie : ')
                #print(l)

                u, v = None, None
                s = 100

                for i in range(len(l) - 1):
                    p = self.graph.get_edge_data(l[i], l[i+1])['prob']
                    #print(p)

                    #print(' {} -> {} ({})'.format(l[i], l[i+1], p))
                    if p < s:
                        s = p
                        u = l[i]
                        v = l[i+1]

                # we remove this edge
                #print('Removing : {} -> {}'.format(u, v))
                self.remove_edge_hand(u, v)

                # for each edge in l, find smallest right_prob

        
        print('{} cycles found'.format(nb_cycles))

        return ineq_series




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

        nb_dual_vars_found = 0

        start_time = time.time()

        if dual_variables_file_name != '':
            
            nb_dual_vars_found = int(percentage_ineq * nb_dual_variables)
            #nb_dual_vars_found = 30

            print('Dual variables : {}'.format(nb_dual_vars_found))


            if add_pairwise_inequalities:

                ineq_graph = IneqGraph(nb_dual_vars_found + 2)

                dual_var_name_to_value = {}

                
                s = random.sample(dual_variables, nb_dual_vars_found)
                
                # sort du plus grand au plus petit
                s.sort(key=lambda pair: pair[VALUE], reverse=True)

                max_diff = abs(s[0][VALUE] - s[-1][VALUE])
                # max_diff : 95% sur
                # <1 diff : 65% sur
                # on create une fonction
                max_odds = 0.999
                min_odds = 0.75
                # odds = a*diff + b

                nb_wrong = 0

                a = (max_odds-min_odds)/max_diff

                ineq_graph.add_node('Source')

                edge_value = -1

                for dual_var in s:

                    dual_var_name_to_value[dual_var[NAME]] = dual_var[VALUE]

                    ineq_graph.add_node(dual_var[NAME])
                    # Source -> VD (0)
                    ineq_graph.add_edge('Source', dual_var[NAME], 0, 1)
                    #ineq_graph.add_edge(dual_var[NAME], 'Sink', -1)

                print(' === ')

                print("Added nodes in {} sec".format(time.time() - start_time))

                ineq_graph.add_node('Sink')
                for dual_var in s:

                    # VD -> Sink (0)
                    ineq_graph.add_edge(dual_var[NAME], 'Sink', 0, 1)

                

                for i in range(len(s) - 1):
                    #print('s[i]({}) = {}'.format(s[i][0], s[i][1]))

                    pi_i_name = s[i][NAME]
                    pi_i_value = s[i][VALUE]

                    #print('{} : {} '.format(pi_i_name, pi_i_value))

                    #print('si > 10 ?? : {}'.format(s[i][1] > 10))
                    for j in range(i + 1, len(s)):

                        pi_j_name = s[j][NAME]
                        pi_j_value = s[j][VALUE]

                        #print('     {} : {} '.format(pi_j_name, pi_j_value))

                        # On check si s[i] >= s[j] ??
                        # On check la difference
                        # On ajoute une probabilité d'erreur

                        # ON AJOUTE ICI DU RANDOM

                        diff = abs(s[i][1] - s[j][1])
                        # on sait que s[i] >= s[j], mais en fonction de la différence, on calcul une probabilité de se tromper
                        #v = random.random() # return a value between [0,1) 1 never there
                        treshold = a*diff + min_odds
                        # PLUS LE TRESHOLD EST HAUT!, PLUS ON EST SUR DE NOTRE INEGALITE (PLUS LA DIFF EST GRANDE)
                        # v = random.uniform(0, 1)
                        v = 0
                        #print('         {}'.format(treshold))   

                        if pi_i_value >= pi_j_value: # AND RANDOM 

                            #print('         i >= j')

                            if v <= treshold:
                                ineq_graph.add_edge(pi_i_name, pi_j_name, edge_value, treshold)
                                #ineq_graph.add_edge(pi_j_name, pi_i_name, 0)
                            else:
                                #print('         but add error')
                                #ineq_graph.add_edge(pi_i_name, pi_j_name, 0)

                                nb_wrong += 1

                                ineq_graph.add_edge(pi_j_name, pi_i_name, edge_value, treshold)

                        else:

                            #print('         j <  i')

                            if v <= treshold:
                                #ineq_graph.add_edge(pi_i_name, pi_j_name, 0)
                                ineq_graph.add_edge(pi_j_name, pi_i_name, edge_value, treshold) 
                            else:

                                nb_wrong += 1
                                # add error
                                #print('        but add error')
                                ineq_graph.add_edge(pi_i_name, pi_j_name, edge_value, treshold)
                                #ineq_graph.add_edge(pi_j_name, pi_i_name, 0) 

                        # tasks_in_new_inequalities.add(pi_1)
                        # tasks_in_new_inequalities.add(pi_2)
                        
                        # # techniquement non, on ajoute pas la, on note les relations
                        # inequalities.append((pi_1, pi_2))

                        # compute odds of having wrong inequality

                print(" === ")
                print('Added Edges in {} secs'.format(time.time() - start_time))

                print('nb wrongs : {}'.format(nb_wrong))

                #ineq_graph.update_csgraph()

                # print(ineq_graph.graph)

                # print('Source : {}'.format(ineq_graph.get_indice_from_node_name('Source')))
                # print('Sink : {}'.format(ineq_graph.get_indice_from_node_name('Sink')))

                # dist_matrix, predecessors = bellman_ford(ineq_graph.graph, directed=True, indices=0, return_predecessors=True)

                # with predecessors, find the dual variables values

                # print(dist_matrix)
                # print(predecessors)

                # ineq_graph.get_longues_ineq_serie()

                # print(s)

                #ineq_graph.bellman_ford()

                #ineq_series = ineq_graph.get_ineq_series()
                

                print('Edges before : {}'.format(ineq_graph.graph.number_of_edges()))

                #nx.draw(ineq_graph.graph, with_labels=True)

                #plt.show()
                
                ineq_series = ineq_graph.get_ineq_series_libr()

                print('Edges after : {}'.format(ineq_graph.graph.number_of_edges()))
                
                

                #ing ineq series:')

                print('{} ineq series of average length : {}'.format(len(ineq_series), sum(len(s) for s in ineq_series) / len(ineq_series)))
                for s in ineq_series:

                    # ICI ON AJOUTE LES SERIES D'INEGALITES

                    # [Source, VD1, ... , VDN, Sink]
                    for i in range(1, len(s) - 2):
                        
                        pi_1 = s[i]
                        pi_2 = s[i+1]

                        tasks_in_new_inequalities.add(pi_1)
                        tasks_in_new_inequalities.add(pi_2)
                        inequalities.append((pi_1, pi_2))


                    #print(s)

                #nx.draw(ineq_graph.graph, with_labels=True)

                #plt.show()

            else:

                # on groupe les variables duales
                s = random.sample(dual_variables, nb_dual_variables)

                # sort du plus grand au plus petit
                s.sort(key = lambda pair: pair[1], reverse=True)

                ineq_groups = []

                max_val = int(s[0][1])
                min_val = int(s[-1][1])

                group_range = 6 # VARIABLE

                for i in range(max_val, min_val, -group_range):
                    
                    group = [x for x in s if (i >= x[1]) and (x[1] > i - group_range)]

                    ineq_groups.append(group)


                    # on les groupe, mais certaines variables duales devraient êtres placés de façon aléatoire (plus de chances d'être
                    # dans un groupe proche)

                print(ineq_groups)
                
                # Groupes sont en ordre du plus grand au plus petit
                ineq_groups = [g for g in ineq_groups if g != []]

                for g, group in enumerate(ineq_groups):

                    for i in range(len(group) - 1 ) : 
                        
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
                        pi_2 = ineq_groups[g + 1][0][0]
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
        
        if percentage_ineq == 0:
            output_file_name += "_default"
        else :
            if add_pairwise_inequalities:
                output_file_name += "_P_{}".format(len(inequalities))
            else:
                output_file_name += "_G_{}_{}".format(len(inequalities), group_range)

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

        


