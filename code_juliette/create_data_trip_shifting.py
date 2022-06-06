from code_juliette.dataset_variables import d_time_table, dict_length, dict_length_hlp, dict_lines, lines, list_depot, list_recharge
from code_juliette.create_data_trip_shifting_variables import seed_list, network_num,  max_minute, rand_delay, var_modif, nb_borne
import random
import math
import os


def create_d_ratio_heures(lines, d_time_table):
    """
    lines : liste de string,
    d_time_table : liste de int,
    retourne un dictionnaire d_ratio_heures[line] qui retourne une liste contenant
    une valeur pour chaque heure. L'heure (max_heure) ayant la plus grande frequence est 1 et
    les heures restantes (x) ont comme valeur (frequence max/ frequence x).
    Si pour certaines heures aucun bus circule, la valeur est 0.
    """

    d_ratio_heures = {}
    max_heure = 0

    for line in lines:
        for freq in d_time_table[line]:
            if freq > max_heure:
                max_heure = freq

    for line in lines:
        d_ratio_heures[line] = [max_heure/x if x != 0 else 0 for x in d_time_table[line]]

    return d_ratio_heures


def create_d_ratio_heures_random_modif(lines, d_time_table, var_modif, seed):
    """
    lines : liste de string,
    d_time_table : liste de int,
    var_modif : int,
    seed : int,
    retourne un dictionnaire d_ratio_heures[line] qui retourne une liste contenant
    une valeur pour chaque heure. L'heure (max_heure) ayant la plus grande frequence est 1 et
    les heures restantes (x) ont comme valeur (frequence max/ frequence x).
    Si pour certaines heures aucun bus circule, la valeur est 0.
    Les frequences sont modifiees par var_modif.
    """

    d_ratio_heures = {}
    max_heure = 0
    d_time_table_modif = {}

    for line in lines:
        random.seed(seed)
        d_time_table_modif[line] = [x + random.randint(-var_modif, var_modif) if x != 0 else x for x in d_time_table[line]]
        d_time_table_modif[line] = [0 if x < 0 else x for x in d_time_table_modif[line]]
    # print(d_time_table_modif)

    for line in lines:
        for freq in d_time_table_modif[line]:
            if freq > max_heure:
                max_heure = freq

    for line in lines:
        d_ratio_heures[line] = [max_heure/x if x != 0 else 0 for x in d_time_table_modif[line]]

    return d_ratio_heures


def create_time_table(lines, max_minute, d_time_table, var_modif=None, seed=None):
    """
    lines : liste de string,
    max_minutes : int
    d_time_table : liste de int,
    var_modif : int,
    seed : int,
    retourne un dictionnaire ayant une liste pour chaque ligne. Cette liste a une
    valeur pour chaque heure egale a la valeur max_minute pour l'heure ayant la
    frequence maximale. Pour le reste, l'entree est egale a la valeur associee a
    l'heure cree par create_d_ratio_heures(lines) * max_minute
    """

    if var_modif is None:
        time_table_ref = create_d_ratio_heures(lines, d_time_table)
    else:
        time_table_ref = create_d_ratio_heures_random_modif(lines, d_time_table, var_modif, seed)

    for line in lines:
        time_table_ref[line] = [round(x * max_minute) for x in time_table_ref[line]]

    return time_table_ref


def create_time_table_minute(lines, max_minute, d_time_table, rand_delay=None, var_modif=None, seed=None):
    """
    lines : liste de string,
    max_minutes : int
    d_time_table : liste de int,
    rand_delay : int,
    var_modif : int,
    seed : int,
    retourne un dictionnaire de time_table et de line_table
    modifie (par le facteur max_minute) pour chaque ligne.
    time_table: pour chaque minute de la journee, indique le nombre de depart a cette minute
    line_table: pour chaque minute de la journee, indique quelle ligne est associe a
    ce depart (dans le meme ordre que celui donne en argument)
    contient les donnes pour plusieurs lignes de bus differents.
    """

    d_time_table_ref = create_time_table(lines, max_minute, d_time_table, var_modif, seed)
    d_time_table_total = {}
    num_trip = 0

    for line in lines:
        if rand_delay is None:
            rand_num = random.randrange(math.floor(max_minute))
        else:
            rand_num = rand_delay
        end_min = rand_num + 60
        flag_sens = 1

        for i in range(23):
            if d_time_table_ref[line][i] != 0:
                start_min = end_min - 60
                for j in range(start_min, 60, d_time_table_ref[line][i]):
                    if flag_sens:
                        d_time_table_total[num_trip] = (i*60 + j, str(int(line)*2))  # (i*60 + j, line)
                        flag_sens = 0
                        num_trip += 1
                    else:
                        d_time_table_total[num_trip] = (i*60 + j, str(int(line)*2 + 1))  # (i*60 + j, line + "R")
                        flag_sens = 1
                        num_trip += 1
                    if i != 22:
                        end_min = j + d_time_table_ref[line][i+1]
                    if end_min < 60:
                        end_min = 60

    return d_time_table_total


def create_voyages_txt(d_time_table, dict_length, dict_length_hlp, dict_lines, seed_num, network_num, network_path):
    """cree le fichier voyages.txt"""
    file = open(network_path + "/voyages.txt", "w")
    for trip in d_time_table:
        file.write("B" + str(trip) + ";" +
                   str(dict_lines[d_time_table[trip][1]][0]) + ";" +
                   str(d_time_table[trip][0]) + ";" +
                   str(dict_lines[d_time_table[trip][1]][-1]) + ";" +
                   str(d_time_table[trip][0] + round(dict_length[d_time_table[trip][1]])) + ";" +
                   d_time_table[trip][1] + ";" + "\n")


def create_hlp_txt(dict_length, dict_length_hlp, dict_lines, seed_num, network_num, network_path):
    """cree le fichier hlp.txt"""
    file = open(network_path + "/hlp.txt", "w")
    for line1 in dict_length_hlp:
        for line2 in dict_length_hlp[line1]:
            file.write(str(dict_lines[line1][-1]) + ";" +
                       str(dict_lines[line2][0]) + ";" +
                       str(round(dict_length_hlp[line1][line2])) + ";" +
                       str(0) + ";" + "\n")


def create_depot_txt(num_depot, seed_num, network_num, network_path):
    """cree le fichier depot.txt"""
    file = open(network_path + "/depots.txt", "w")
    for depot in num_depot:
        file.write(str(depot) + ";" +
                   str(200) + ";" + "\n")

def create_recharge_txt(num_borne, seed_num, network_num, nb_borne, network_path):
    file = open(network_path + "/recharge.txt", "w")
    for depot in num_borne:
        file.write(str(depot) + ";" +
                   str(nb_borne) + ";" + "\n")


def create_network_data(seed_list, network_num, lines, max_minute, list_depot, d_time_table, list_recharge, 
        nb_borne, dict_length, dict_length_hlp, dict_lines, rand_delay=None, var_modif=None):
    for seed_num in seed_list:
        network_path = "Networks/Network" + str(network_num) + '_' + str(max_minute).replace('.', 'p') +  "_" + str(seed_num)  
        if not os.path.exists(network_path):
            os.mkdir(network_path)
        d_time_table_created = create_time_table_minute(lines, max_minute, d_time_table, rand_delay, var_modif, seed_num)
        create_voyages_txt(d_time_table_created, dict_length, dict_length_hlp, dict_lines, seed_num, network_num, network_path)
        create_hlp_txt(dict_length, dict_length_hlp, dict_lines, seed_num, network_num, network_path)
        create_depot_txt(list_depot, seed_num, network_num, network_path)
        create_recharge_txt(list_recharge, seed_num, network_num, nb_borne, network_path)


# create_network_data(seed_list, network_num, lines, max_minute, list_depot, d_time_table, list_recharge, nb_borne, 
#     dict_length, dict_length_hlp, dict_lines, rand_delay, var_modif)
