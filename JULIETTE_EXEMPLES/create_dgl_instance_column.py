import pandas as pd
from math import ceil

list_pb = ['4a_11_'+str(k) for k in range(10)]
cout_fixe=1000
nb_veh=35
sigma_max=100000
vitesse=18/60
e_km=1050 
e_min=11000/60
cout_w=2
cout_t=4
delta=45
recharge=15

out_file = open("out", "w")
out_file.close()

for i, pb in enumerate(list_pb):
    with open('out', 'a') as out:
        out.write(f'{pb} ({i+1}/{len(list_pb)})\n')
    df_nodes = pd.DataFrame(columns=['name','t_s','t_e','n_s','n_e'])
    

    folder_name = "Networks/Network" + pb

    trip_file = open(folder_name + "/voyages.txt", "r")
    trip_list = trip_file.readlines()
    nb_trip = len(trip_list)

    depot_file = open(folder_name + "/depots.txt", "r")
    depot_list = depot_file.readlines()
    nb_depot = len(depot_list)

    hlp_file = open(folder_name + "/hlp.txt", "r")
    hlp = {}
    for line in hlp_file.readlines():
        trip = line.split(';')
        if trip[0] in hlp:
            hlp[trip[0]][trip[1]] = int(trip[2])
        else :
            hlp[trip[0]] = {trip[1] : int(trip[2])}
    recharge_file = open(folder_name + "/recharge.txt", "r")
    recharge_list = recharge_file.readlines()
    nb_recharge = len(recharge_list)

    t_min = 1400 #
    t_max = 0 #
    for trip in trip_list:
        trip_end = int(trip.split(';')[4])
        if trip_end < t_min:
            t_min = trip_end
        if trip_end > t_max:
            t_max = trip_end
    t_min = (t_min//recharge)*recharge
    t_max = int(ceil(t_max/recharge)+4)*recharge
    periode = [i for i in range(t_min, t_max+recharge, recharge)]
    P = len(periode) # nombre de périodes de temps

    def c_trip(n1,n2,f1,d2): # cout du trip de n1 à n2 entre un trajet finissant à n1 à f1
        # et commençant à d2 en n2
        t = hlp[n1][n2]
        c = cout_t*t + cout_w*(d2-f1-t)
        return c, t, d2-f1-t, d2-f1

    def e_trip(n_d,n_f,d,f): # consommation d'énergie sur le trip de n_d à n_f entre d et f
        t_vide = hlp[n_d][n_f] # temps du voyage à vide
        dist =  vitesse*t_vide # distance parcourrue
        e = e_km*dist + e_min*(int(f)-int(d)-t_vide) # l'énergie consommée est la somme de l'énergie consommée par la distance
        # et de l'énergie consommée à l'arrêt
        return e

    def e_hlp(n1, n2): # consommation d'énergie sur le trip à vide de n1 à n2
        t_vide = hlp[n1][n2] # temps du voyage à vide
        dist =  vitesse*t_vide # distance parcourrue
        e = e_km*dist # énergie consommée
        return e

    def c_depot(n1,n2):
        t = hlp[n1][n2]
        c = cout_t*t
        return c, t, 0

    def compatible(n1,n2,f1,d2):
        t = hlp[n1][n2]
        if (d2-f1-t)>=0 and (d2-f1)<=delta:
            return True
        else :
            return False

    # Nodes
    for depot, depot_info in enumerate(depot_list):
        d = depot_info.split(";")
        df_nodes.loc[len(df_nodes)] = ["o_D" + str(depot), 0, 0, d[0], d[0]]
        df_nodes.loc[len(df_nodes)] = ["k_D" + str(depot), t_max, t_max, d[0], d[0]]
    for trip, info_trip  in enumerate(trip_list):
        t1 = info_trip.split(";")
        df_nodes.loc[len(df_nodes)] = ["n_T" + str(trip), t1[2],t1[4], t1[1], t1[3]]
    for borne, borne_info in enumerate(recharge_list):
        b = borne_info.split(";")
        for t in range(P):
            df_nodes.loc[len(df_nodes)] =["w_H" + str(borne) + "_P" + str(t), periode[t], periode[t], b[0], b[0]]
            df_nodes.loc[len(df_nodes)] =["c_H" + str(borne) + "_P" + str(t), periode[t], periode[t], b[0], b[0]]

    list_edges = []
    # Arcs 
    for i, trip1 in enumerate(trip_list):
        t1 = trip1.split(";")

        for k, depot in enumerate(depot_list):
            d = depot.split(";")
            e = int(e_hlp(d[0],t1[1]))
            c, t, w = c_depot(d[0],t1[1])
            list_edges.append(["o_D" + str(k), "n_T" + str(i), c, e, t, w, t, 0, t1[5]])
            e = int(e_trip(t1[1],t1[3],t1[2],t1[4])+e_hlp(t1[3],d[0]))
            c, t, w = c_depot(t1[3],d[0])
            list_edges.append(["n_T" + str(i), "k_D" + str(k), c, e, t, w, t, t1[5], 0])

        for j, trip2 in enumerate(trip_list):
            t2 = trip2.split(";")
            if compatible(t1[3],t2[1],int(t1[4]),int(t2[2])):
                e = int(e_trip(t1[1],t1[3],t1[2],t1[4])+e_hlp(t1[3],t2[1]))
                c, t, w, d = c_trip(t1[3],t2[1],int(t1[4]),int(t2[2]))
                list_edges.append(["n_T" + str(i), "n_T" + str(j), c, e, t, w, d, t1[5], t2[5]])

        for h, borne in enumerate(recharge_list):
            b = borne.split(";")
            t = hlp[t1[3]][b[0]]
            p = min([i for i in range(P) if periode[i]>=int(t1[4])+t]+[P+1])
            if p != P+1:
                e = int(e_trip(t1[1],t1[3],t1[2],t1[4])+e_hlp(t1[3],b[0]))
                c, t, w = c_depot(t1[3],b[0])
                list_edges.append(["n_T" + str(i), "w_H" + str(h) + "_P" + str(p), c, e, t, w, 
                    periode[p]-int(t1[4]), t1[5], 0])

            t = hlp[b[0]][t1[1]]
            p = max([i for i in range(P) if periode[i]<=int(t1[2])-t]+[-1])
            if p != -1 :
                e = int(e_hlp(b[0],t1[1]))
                c, t, w = c_depot(b[0],t1[1])
                list_edges.append(["w_H" + str(h) + "_P" + str(p),"n_T" + str(i), c, e, t, w, 
                    int(t1[2])-periode[p], 0, t2[5]])
             
    for h, borne in enumerate(recharge_list):
        b = borne.split(";")
        for p in range(P-1):
            list_edges.append(["w_H" + str(h) + "_P" + str(p), "w_H" + str(h) + "_P" + str(p+1),
                0, 0, 0, 0, recharge, 0, 0])

            list_edges.append(["w_H" + str(h) + "_P" + str(p), "c_H" + str(h) + "_P" + str(p+1),
                0, 0, 0, 0, recharge, 0, 0])

            if p >= 1:
                list_edges.append(["c_H" + str(h) + "_P" + str(p), "c_H" + str(h) + "_P" + str(p+1),
                0, 0, 0, 0, recharge, 0, 0])

                list_edges.append(["c_H" + str(h) + "_P" + str(p), "w_H" + str(h) + "_P" + str(p),
                0, 0, 0, 0, 0, 0, 0])

        list_edges.append(["c_H" + str(h) + "_P" + str(P-1), "w_H" + str(h) + "_P" + str(P-1),
            0, 0, 0, 0, 0, 0, 0])

        for k, depot in enumerate(depot_list):
            d = depot.split(";")
            e = int(e_hlp(b[0],d[0]))
            c, t, w = c_depot(b[0],d[0])
            list_edges.append(["w_H" + str(h) + "_P" + str(P-1), "k_D" + str(k), c, e, t, w, t, 0, 0])
        
    df_edges = pd.DataFrame(list_edges, columns=['src','dst','c','r1','t_d','t_w','delta', 'line_i', 'line_j'])
    df_nodes['type'] = df_nodes['name'].apply(lambda x:x.split('_')[0])
    for lettre in ['o','k','n','w','c','d']:
        df_nodes[lettre] = (df_nodes['type']==lettre).astype(int)

    name_idxn = dict(df_nodes.reset_index().set_index('name')['index'])

    df_edges['idx_src'] = df_edges['src'].apply(lambda x : name_idxn[x])
    df_edges['idx_dst'] = df_edges['dst'].apply(lambda x : name_idxn[x])
    df_edges = df_edges.astype({'c': float, 'r1': float})

    df_edges['c_stand'] = df_edges['c']/df_edges['c'].max()
    df_edges['r1_stand'] = df_edges['r1']/sigma_max

    df_edges['sol_int'] = 0
    df_edges['LR'] = 0
    df_edges['RP'] = 0
    df_edges['CG'] = 0
    try :
        with open("arcSolvingInfoProblem" + pb + ".out", "r") as sol_file :
            sol_lines = sol_file.readlines()
            nb_ligne = len(sol_lines)
            for i in range(nb_ligne-1):
                infos = sol_lines[i].rstrip().split(' ')
                df_edges.loc[(df_edges['src'] == infos[0])&(df_edges['dst'] == infos[1]),['sol_int','LR','RP','CG']] = infos[2:]
    except :
        pass 

    hlp = pd.read_csv('Networks/Network'+pb+'/hlp.txt',sep=';',header=None, usecols=[0,1,2], names=['n1','n2','t'])
    dict_nodes = dict()
    for x in hlp[hlp.t==0].itertuples():
        if x.n1 not in dict_nodes.keys():
            dict_nodes[x.n1] = [x.n2]
        else :
            if x.n2 not in dict_nodes[x.n1]:
                dict_nodes[x.n1].append(x.n2)

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

    df_nodes['n_s'] = df_nodes['n_s'].apply(lambda x:dict_nodes_2[x])
    df_nodes['n_e'] = df_nodes['n_e'].apply(lambda x:dict_nodes_2[x])
    for noeud in df_nodes['n_s'].unique():
        df_nodes['s_'+noeud] = (df_nodes['n_s']==noeud).astype(int)
    for noeud in df_nodes['n_e'].unique():
        df_nodes['e_'+noeud] = (df_nodes['n_e']==noeud).astype(int)
        
    df_trip = pd.read_csv('Networks/Network'+pb+'/voyages.txt',sep=';',header=None,usecols=[0,1,2,3,4,5],
        names=['trip','n_d','t_d','n_f','t_f','line'])
    df_trip['n_d'] = df_trip['n_d'].apply(lambda x:dict_nodes_2[x])
    df_trip['n_f'] = df_trip['n_f'].apply(lambda x:dict_nodes_2[x])

    df_nodes = df_nodes.astype({'t_s': float, 't_e': float})

    df_nodes['nb_dep_10'] = df_nodes.apply(lambda x:len(df_trip.loc[(df_trip.n_d==x.n_s)&(df_trip.t_d>=x.t_s)&
        (df_trip.t_d<=x.t_s+10)])-1 if x.n==1 else 0, axis=1)
    df_nodes['nb_dep_id_10'] = df_nodes.apply(lambda x:len(df_trip.loc[(df_trip.n_d==x.n_s)&(df_trip.n_f==x.n_e)&
        (df_trip.t_d>=x.t_s)&(df_trip.t_d<=x.t_s+10)])-1 if x.n==1 else 0, axis=1)
    df_nodes['nb_fin_10'] = df_nodes.apply(lambda x:len(df_trip.loc[(df_trip.n_f==x.n_e)&(df_trip.t_f<=x.t_e)&
        (df_trip.t_f>=x.t_e-10)])-1 if x.n==1 else 0, axis=1)
    df_nodes['nb_fin_id_10'] = df_nodes.apply(lambda x:len(df_trip.loc[(df_trip.n_f==x.n_e)&(df_trip.t_f<=x.t_e)&
        (df_trip.n_d==x.n_s)&(df_trip.t_f>=x.t_e-10)])-1 if x.n==1 else 0, axis=1)

    # df_edges = df_edges.astype({'line_i': int, 'line_j': int})
    # lines = df_edges['line_i'].unique()

    df_edges = pd.merge(df_edges, df_nodes[['name','t_e', 'n_e',]], left_on = ['src'], right_on = ['name'])
    df_edges = pd.merge(df_edges, df_nodes[['name','t_s', 'n_s']], left_on = ['dst'], right_on = ['name'])

    df_edges['rg_id'] = df_edges.groupby(['src','n_s'])["t_s"].rank("dense", ascending=True)
    df_edges['rg'] = df_edges.groupby(['src'])["t_s"].rank("dense", ascending=True)

    # df_edges.rename(columns= {'c_x': 'c', 'c_y': 'c_x', 'c':'c_y'}, inplace = True)
    del df_edges['name_x']
    del df_edges['name_y']

    df_nodes.to_csv('DGLgraphs_column/nodes'+pb+'.csv', sep = '\t')
    df_edges.to_csv('DGLgraphs_column/edges'+pb+'.csv', sep = '\t')

