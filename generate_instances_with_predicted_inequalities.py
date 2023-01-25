import pandas as pd
import os
import networkx as nx
from glob import glob
import sys

from math import ceil

from networkx import bellman_ford_path, find_cycle, NetworkXNoCycle, has_path

from instances_params import *

# given a folder name (for now)
# which has all the network information
# create a new gencol input file with the predicted inequalities

class IneqGraph:

    def __init__(self):

        self.indice_to_node_name = []
        self.node_name_to_indice = {}

        self.graph = nx.DiGraph()

        self.degrees = {}

    def add_node(self, node_name):

        self.node_name_to_indice[node_name] = len(self.indice_to_node_name)
        self.indice_to_node_name.append(node_name)
        
        self.graph.add_node(node_name)

    def add_edge(self, from_node_name, to_node_name, value, prob_right):

        self.graph.add_weighted_edges_from([(from_node_name, to_node_name, value)], 'weight', prob=prob_right)
        
    def get_indice_from_node_name(self, node_name):

        return self.node_name_to_indice[node_name]

    def get_node_name_from_indice(self, indice):

        return self.indice_to_node_name[indice]

    def establish_degrees(self):

        for n in self.graph.nodes():

            self.degrees[n] = self.graph.in_degree(n) - self.graph.out_degree(n)

    def validate_edges(self):

        for e in list(self.graph.edges()):
            
            deg_u = self.degrees[e[0]]
            deg_v = self.degrees[e[1]]

            if deg_v < deg_u:
                self.graph.remove_edge(e[0], e[1])

    def remove_cycles(self):

        while(True):
            try:
                cycle = find_cycle(self.graph, source='Source')
                self.graph.remove_edges_from(cycle)
                
            except NetworkXNoCycle:
                break

    def bellman_ford(self):

        path = bellman_ford_path(self.graph, source='Source', target='Sink')

        return False, path

    def get_ineq_series(self, nb_series=10):
        ineq_series = []
        while(True):
            
            has_neg, path = self.bellman_ford()
            if not has_neg:
                if len(path) <= 3:
                    break
                serie = path[1:-1]
                for i in range(len(serie) - 1):
                    u = serie[i]
                    v = serie[i+1]
                    self.graph.remove_edge(u,v)
                ineq_series.append(serie)

                if len(ineq_series) >= nb_series:
                    break

        return ineq_series

def create_file(instance_id, out_suffix, l_tresh, u_tresh, min_coef, max_coef, max_borne):

    network_folder = 'Networks/Network{}'.format(instance_id)

    # check if have predictions : 
    if not os.path.exists('{}/inequalities_predictions.csv'.format(network_folder)):
        return

    trips_file = open(network_folder + "/voyages.txt", "r")
    trips_list = trips_file.readlines()
    nb_trip = len(trips_list)

    depot_file = open(network_folder + '/depots.txt', 'r')
    depots_list = depot_file.readlines()
    nb_depots = len(depots_list)

    chrg_stations_file = open(network_folder + '/recharge.txt', 'r')
    chrg_stations_list = chrg_stations_file.readlines()
    nb_charg_stations = len(chrg_stations_list)

    df_predictions = pd.read_csv(network_folder + '/inequalities_predictions.csv', index_col=[0])
    #df_predictions.drop(columns=['0'], inplace=True)
    # 'A', 'B', 'pred', 'real', 'A_pi', 'B_pi'

    tasks_in_new_inequalities = set()
    inequalities = []

    # On fait le graphe d'inégalités pairwise selon nos prédictions

    nodes = df_predictions['A'].unique()
    nodes = [x.replace('n','Cover') for x in nodes]

    ineq_graph = IneqGraph()

    #print("Adding basic nodes and edges")

    # Source -> n_i
    ineq_graph.add_node('Source')
    edge_value = -1
    for n in nodes:
        ineq_graph.add_node(n)
        ineq_graph.add_edge('Source', n, 0, 1)
    
    # n_i -> Sink
    ineq_graph.add_node('Sink')
    for n in nodes:
        ineq_graph.add_edge(n, 'Sink', 0, 1)

    #print("Adding pairwise inequalities edges")

    pi_vals = df_predictions.set_index('A')['A_pi'].to_dict()

    # Pairwise inequalities
    df_pairwise_inequalities = df_predictions[df_predictions['pred'] > u_tresh]
    df_pairwise_inequalities.loc[:, 'A_B'] = df_pairwise_inequalities['A'] + '-' + df_pairwise_inequalities['B']
    ineq_pairs = list(df_pairwise_inequalities['A_B'].unique())

    pair_probs = df_pairwise_inequalities.set_index('A_B')['pred'].to_dict()

    #print('predictions : {}'.format(df_pairwise_inequalities.shape[0]))

    good_ineq = 0
    added_ineq = 0

    found_ineq = set()
    ineq_probs = dict()

    for p in ineq_pairs:
        #pi_i >= pi_j
        A, B = p.split('-')
        Cover_A = A.replace('n', 'Cover')
        Cover_B = B.replace('n', 'Cover')

        found_ineq.add(p)

        prob = pair_probs[p]

        A_real_val = pi_vals[A]
        B_real_val = pi_vals[B]

        if A_real_val >= B_real_val:
            good_ineq += 1
        added_ineq += 1

            ### JUSTE POUR TESTER, ICI ON MET JUSTE DES VRAIS INEG.

        key = Cover_A + '-' + Cover_B
        ineq_probs[key] = prob
    
        ineq_graph.add_edge(Cover_A, Cover_B, edge_value, p)

    df_pairwise_zeroes = df_predictions[df_predictions['pred'] < l_tresh] # Ceux qu'on est SUR qui sont =0
    df_pairwise_zeroes.loc[:,'A_B'] = df_pairwise_zeroes['A'] + '-' + df_pairwise_zeroes['B']
    ineq_pairs = set(df_pairwise_zeroes['A_B'].unique())
    pair_probs = df_pairwise_zeroes.set_index('A_B')['pred'].to_dict()

    # pour les paires A,B t.q. pred(A,B) < 0.35, on ajoute B->A
    for p in ineq_pairs:
        A, B = p.split('-')
        rev = B + '-' + A

        if A == B:
            continue

        # MAINTENANT : 
        if rev not in found_ineq:

            # on l'ajoute direct

            A_real_val = pi_vals[A]
            B_real_val = pi_vals[B]

            if B_real_val >= A_real_val:
                good_ineq += 1

                ### JUSTE POUR TESTER, ICI ON MET JUSTE DES VRAIS PREDS
            found_ineq.add(rev)

            Cover_A = A.replace('n', 'Cover')
            Cover_B = B.replace('n', 'Cover')

            prob = 1 - pair_probs[p]
            key = Cover_B + '-' + Cover_A
            ineq_probs[key] = prob

            ineq_graph.add_edge(Cover_B, Cover_A, edge_value, prob)

            # added_ineq += 1

        # AVANT :
        # if (rev not in found) and (A != B) and (rev in possible_pairs):
        #     found.add(p)
        #     Cover_A = A.replace('n', 'Cover')
        #     Cover_B = B.replace('n', 'Cover')

        #     A_real_val = df_pairwise_zeroes.loc[df_pairwise_zeroes['A_B'] == p, 'A_pi'].values[0]
        #     B_real_val = df_pairwise_zeroes.loc[df_pairwise_zeroes['A_B'] == p, 'B_pi'].values[0]

        #     if A_real_val >= B_real_val:
        #         good_ineq += 1

        #     prob = df_pairwise_zeroes.loc[df_pairwise_zeroes['A_B'] == p, 'pred'].values[0]

        #     key = '{}-{}'.format(Cover_A, Cover_B)
        #     ineq_probs[key] = prob

        #     # On doit trouver la valeur de la probabilite P(A,B)
        #     added_ineq += 1
        #     ineq_graph.add_edge(Cover_A, Cover_B, edge_value, prob)

    # Juste pour tester
    #return

    #print('Good ineq added : {}'.format(good_ineq/added_ineq))

    # print("Establishing degrees")
    
    # Establishing degrees
    ineq_graph.establish_degrees()

    # print("Validating edges")

    # Validating node degrees
    ineq_graph.validate_edges()

    # print("Removing cycles")

    # Removing cycles
    ineq_graph.remove_cycles()

    # print("Getting ineq series")

    #print('Edges : {}'.format(ineq_graph.graph.number_of_edges()))
    #print('Nodes : {}'.format(ineq_graph.graph.number_of_nodes()))

    # Getting series ineq
    ineq_series = ineq_graph.get_ineq_series()

    # # Verifie l'accuracy des inégalités
    # print('Counting')
    # nb_good_1 = len(df_predictions[(df_predictions.real == 1.0) & (df_predictions.pred == 1.0)])
    # nb_good_0 = len(df_predictions[(df_predictions.real == 0.0) & (df_predictions.pred == 0.0)])
    # tot_good = nb_good_0 + nb_good_1
    # tot_rows = df_predictions.shape[0]
    # model_acc = tot_good/tot_rows
    # #model_acc = df_predictions[df_predictions['pred'] == df_predictions['real']].sum() / df_predictions.shape[0]

    # nb_ineq = 0
    # nb_good = 0

    # pi_values = df_predictions.drop_duplicates(subset=['A'])

    # for serie in ineq_series:

    #     ineq = []

    #     for i in range(len(serie) - 1):

    #         A = serie[i].replace('Cover_', 'n_')
    #         B = serie[i+1].replace('Cover_', 'n_')

    #         A_pi = pi_values.loc[pi_values['A'] == A, 'A_pi'].values[0]
    #         B_pi = pi_values.loc[pi_values['A'] == B, 'A_pi'].values[0]

    #         ineq.append((A, A_pi))

    #         if A_pi >= B_pi:
    #             nb_good += 1
            
    #         nb_ineq += 1
        
    #     ineq.append((B, B_pi))

    #     print(ineq)

    # after_review_acc = nb_good / nb_ineq

    # print('Before : {} - After : {}'.format(model_acc, after_review_acc))

    # Dessine un graph
    #print(ineq_series)
    wrong_ineq = 0
    total_ineq = 0
    for s in ineq_series:

        for i in range(len(s) - 1):

            pi_i = s[i]
            pi_j = s[i+1]

            # ici sont n_{}, mais dans le graphe sont : Cover_{}
            # u = pi_i.replace('Cover', 'n')
            # v = pi_j.replace('Cover', 'n')

            key = '{}-{}'.format(pi_i, pi_j)
            pred_prob = ineq_probs[key]

            # given a pre_prob, add a coef. weight : more sure, less weight, less sure more weight:
            # 0.5 = 50% sure, it's the worst
            # 1 or 0 = 100% sure
            if min_coef > 0 and max_coef > 0:
                sure_level = 0.5 + abs(pred_prob - 0.5)
                coef = (1 - sure_level)/0.5 * max_coef
                coef = max(coef, min_coef) # so min coef = 0.1
            else:
                coef = 0
            # No coef. impact
            #coef = 0

            # sure = 95 : 0.1 * 15 = 1.5
            # sure = 75 :          = 7.5
            A = pi_i.replace('Cover', 'n')
            B = pi_j.replace('Cover', 'n')
            A_val = pi_vals[A]
            B_val = pi_vals[B]
            if A_val < B_val:
                wrong_ineq += 1
            total_ineq += 1

            
            tasks_in_new_inequalities.add(pi_i)
            tasks_in_new_inequalities.add(pi_j)

            inequalities.append((pi_i, pi_j, coef))
    #print('{}/{} WRONG INEQ ({})'.format(wrong_ineq, total_ineq, wrong_ineq/total_ineq))

    # print("Writing output file")

    output_file_path = network_folder

    output_file_name = "inputProblem{}_{}_P_{}_inequalities".format(instance_id, len(inequalities), out_suffix)

    
    output_file = open('{}/{}.in'.format(network_folder, output_file_name), "w")
    

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

        # if on met le coef
        coef = ineq[2]
        # else
        # coef = 0

        if max_borne > 0:
            borne = '[0 {}]'.format(max_borne)
        else:
            borne = ''

        cols_string += "Y_{} {} {} ({} -1) ({} 1);\n".format(i, coef, borne, pi_1, pi_2)
        #cols_string += "Y_" + str(i) + " 0 (" + pi_1 + " -1) (" + pi_2 + " 1);\n"
    
    
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

if __name__ == '__main__':

    #nb_instances = len(glob('Networks/*')) - 1
    # Pour la validation
    nb_instances = 100

    if len(sys.argv) <= 1:
        print('Missing arguments : suffix for output, lower_tresh, upper_tresh, min_coef (0 if no coef), max_coef (0 if no coef), max_borne 0 if no borne),')
        sys.exit()

    out_suffix = sys.argv[1]
    l_tresh = float(sys.argv[2])
    u_tresh = float(sys.argv[3])
    min_coef = float(sys.argv[4])
    max_coef = float(sys.argv[5])
    max_borne = float(sys.argv[6])

    nb = 1
    for instance in glob('Networks/*'):

        if '.pkl' in instance:
            continue

        instance_id = instance.replace('Networks/Network','')
        instance_seed = int(instance_id.split('_')[-1])
        if instance_seed < 175:
            continue

        create_file(instance_id, out_suffix, l_tresh, u_tresh, min_coef, max_coef, max_borne)

        print('{}/{} done : {}'.format(nb, nb_instances, instance_id))
        nb += 1
