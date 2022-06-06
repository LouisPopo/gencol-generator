from dis import dis
from math import ceil
# Paramètres
list_pb = ["4a_0_0"]

import time
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))

cout_fixe = 1000 # cout d'un véhicule
nb_veh = 35 # nombre de véhicules diponibles à un dépot

sigma_max = 363000 # en Wh
vitesse = 18/60 # en km/min (en moyenne sans compter les arrêts)
e_km = 1050 # énergie consommée en Wh par km parcourru
e_min = 11000/60 # énergie consommée en Wh par minute passée à l'arrêt
#energie_min = int(sigma_max/(8*60)) # énergie consommée en Wh 
                # par minute de trajet
cout_w = 2 # cout d'attente par minute
cout_t = 4 # cout voyage à vide par minute
delta = 45 # temps maximum entre la fin d'un trajet et le début d'un autre

p = 15 # période d'échantillonage pour la recharge
#rho = int(50000*p/60) # duree d'un arc de recharge (chargeur de 50kW)

def create_input(list_pb, cout_fixe=1000, nb_veh=35, sigma_max=363000, vitesse=18/60, e_km=1050, 
    e_min=11000/60, cout_w=2, cout_t=4, delta=45, recharge=15) :
    for pb in list_pb :
        folder_name = "Networks/Network" + pb

        input_file = open("Input/testinputProblem" + pb + ".in", "w")
        trip_file = open(folder_name + "/voyages.txt", "r")
        trip_list = trip_file.readlines()
        nb_trip = len(trip_list)

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


        def c_trip(n1,n2,f1,d2): # cout du trip de n1 à n2 entre un trajet finissant à n1 à f1
            # et commençant à d2 en n2
            t = hlp[n1][n2]
            c = cout_t*t + cout_w*(d2-f1-t)
            return c

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
            return c

        def compatible(n1,n2,f1,d2):
            t = hlp[n1][n2]
            if (d2-f1-t)>=0 and (d2-f1)<=delta:
                return True
            else :
                return False

        print("--- %s seconds ---" % (time.time() - start_time))

        # Rows
        input_file.write("Rows = {\n")

        for trip in range(nb_trip):
            input_file.write("Cover_T" + str(trip) + " = 1 TaskStrong;\n")
        for depot in range(nb_depot):
            input_file.write("Count_D" + str(depot) + " = 0;\n")
        for i, borne in enumerate(recharge_list):
            nb_borne = borne.split(";")[1]
            for t in range(P):
                input_file.write("Max_H" + str(i) + "_P" + str(t) + " <= " + str(nb_borne) + ";\n")
        input_file.write("};\n\n")

        print("--- %s seconds ---" % (time.time() - start_time))

        # Tasks
        input_file.write("Tasks = {\n")
        for trip in range(nb_trip):
            input_file.write("t_T" + str(trip) + " Cover_T" + str(trip) +  
            " Strong;\n")
        for depot in range(nb_depot):
            input_file.write("t_D" + str(depot) + "o Weak;\n")
            input_file.write("t_D" + str(depot) + "k Weak;\n")
        input_file.write("};\n\n")

        print("--- %s seconds ---" % (time.time() - start_time))


        # Columns 
        input_file.write("Columns = {\n")
        for depot in range(nb_depot):
            input_file.write("Veh_D" + str(depot) + " " + str(cout_fixe) +
            " Int [0 " + str(nb_veh) + "] (Count_D" + str(depot) + " -1);\n")
        input_file.write("};\n\n")

        print("--- %s seconds ---" % (time.time() - start_time))

        # Ressources
        input_file.write("Resources = {\nr_SoC_Inv;\nr_Rch;\nr_Not_Rch;\n};\n\n")

        # Nodes
        input_file.write("Nodes = {\n")
        for depot in range(nb_depot):
            input_file.write("o_D" + str(depot) + " [0 0] [0 0] [0 0] " + 
            "t_D" + str(depot) + "o (Count_D" + str(depot) + " 1);\n")
            input_file.write("k_D" + str(depot) + " [0 " + str(sigma_max) +
            "] [0 1] [0 1] " + "t_D" + str(depot) + "k;\n")
        for trip  in range(nb_trip):
            input_file.write("n_T" + str(trip) + " [0 " + str(sigma_max) +
            "] [0 0] [0 0] " + "t_T" + str(trip) + ";\n")
        for borne in range(nb_recharge):
            for t in range(P):
                input_file.write("w_H" + str(borne) + "_P" + str(t) + " [0 " + 
                str(sigma_max) +"] [0 1] [0 1];\n")
                input_file.write("c_H" + str(borne) + "_P" + str(t) + " [0 " + 
                str(sigma_max) +"] [1 1] [0 1];\n")
        input_file.write("};\n\n")

        print("--- %s seconds ---" % (time.time() - start_time))

        # Arcs 
        input_file.write("Arcs = {\n")
        for i, trip1 in enumerate(trip_list):
            t1 = trip1.split(";")

            for k, depot in enumerate(depot_list):
                d = depot.split(";")
                e = int(e_hlp(d[0],t1[1]))
                c = c_depot(d[0],t1[1])
                input_file.write("o_D" + str(k) + " n_T" + str(i) + " " + str(c)
                + " [" + str(e) + " 0 0];\n")
                e = int(e_trip(t1[1],t1[3],t1[2],t1[4])+e_hlp(t1[3],d[0]))
                c = c_depot(t1[3],d[0])
                input_file.write("n_T" + str(i) + " k_D" + str(k) + " " + str(c)
                + " [" + str(e) + " 0 0];\n")

            for j, trip2 in enumerate(trip_list):
                t2 = trip2.split(";")
                if compatible(t1[3],t2[1],int(t1[4]),int(t2[2])):
                    e = int(e_trip(t1[1],t1[3],t1[2],t1[4])+e_hlp(t1[3],t2[1]))
                    c = c_trip(t1[3],t2[1],int(t1[4]),int(t2[2]))
                    input_file.write("n_T" + str(i) + " n_T" + str(j) + " " + str(c)
                + " [" + str(e) + " 0 0];\n")
            
            for h, borne in enumerate(recharge_list):
                b = borne.split(";")
                t = hlp[t1[3]][b[0]]
                p = min([i for i in range(P) if periode[i]>=int(t1[4])+t]+[P+1])
                if p != P+1:
                    e = int(e_trip(t1[1],t1[3],t1[2],t1[4])+e_hlp(t1[3],b[0]))
                    c = c_depot(t1[3],b[0])
                    input_file.write("n_T" + str(i) + " w_H" + str(h) + "_P" + str(p) + " " + str(c)
                    + " [" + str(e) + " 0 1];\n")
                
                t = hlp[b[0]][t1[1]]
                p = max([i for i in range(P) if periode[i]<=int(t1[2])-t]+[-1])
                if p != -1 :
                    e = int(e_hlp(b[0],t1[1]))
                    c = c_depot(b[0],t1[1])
                    input_file.write("w_H" + str(h) + "_P" + str(p) + " n_T" + str(i) + " " + str(c)
                    + " [" + str(e) + " -1 0];\n")
                    

        print("--- %s seconds ---" % (time.time() - start_time))

        for h, borne in enumerate(recharge_list):
            b = borne.split(";")
            for p in range(P-1):
                input_file.write("w_H" + str(h) + "_P" + str(p) + " w_H" + str(h) + "_P" + str(p+1) +
                " 0 [0 0 0];\n")

                input_file.write("w_H" + str(h) + "_P" + str(p) + " c_H" + str(h) + "_P" + str(p+1) +
                " 0 [-" + str(recharge) + " 1 -1] (Max_H" + str(h) + "_P" + str(p) + " 1);\n")

                if p >= 1:
                    input_file.write("c_H" + str(h) + "_P" + str(p) + " c_H" + str(h) + "_P" + str(p+1) +
                    " 0 [-" + str(recharge) + " 0 0] (Max_H" + str(h) + "_P" + str(p) + " 1);\n")

                    input_file.write("c_H" + str(h) + "_P" + str(p) + " w_H" + str(h) + "_P" + str(p) +
                    " 0 [0 0 0];\n")

            input_file.write("c_H" + str(h) + "_P" + str(P-1) + " w_H" + str(h) + "_P" + str(P-1) +
            " 0 [0 0 0];\n")

            for k, depot in enumerate(depot_list):
                d = depot.split(";")
                e = int(e_hlp(b[0],d[0]))
                c = c_depot(b[0],d[0])
                input_file.write("w_H" + str(h) + "_P" + str(P-1) + " k_D" + str(k) + " " + str(c) +
                " [" + str(e) + " 0 0];\n")

        input_file.write("};\n\n")

        print("--- %s seconds ---" % (time.time() - start_time))

        # Networks 
        input_file.write("Networks = {\n")
        for depot in range(nb_depot):
            input_file.write("net_D" + str(depot) + " o_D" + 
            str(depot) + " (k_D" + str(depot) + ");\n")
        input_file.write("};")
