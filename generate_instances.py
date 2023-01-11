# Une instance = une configuration d'un réseau avec un certain nombres de trips, un nombre de bus et de bornes (?), etc. 

# Pour l'instant on utilise seulement un dataset de réseau : 4b

# SCRIPT : va generer un certain nombre d'instances, avec certaines configurations différentes

# Essai 1 : (Selon Juliette), on joue seulement avec max_min et seeds. var_modif, nb_bornes et nb_veh sont fixes

# Nomenclature des Folder networks : Network{network_num}_{max_min}_{seed}/[depots,hlp,recharge,voyages].txt


# PARAMETRES AVEC UN IMPACT : network_num, max_minute, seed_list, nb_borne

from code_juliette.create_data_trip_shifting import create_network_data

from instances_params import *

from code_juliette.dataset_trip_shifting.dataset4b_variables import d_time_table, dict_length, dict_length_hlp, dict_lines, lines, list_depot, list_recharge

nb_instances_per_max_min = 25 # taille

for max_minute in max_minutes:
    seed_list = [(175 + x) for x in range(nb_instances_per_max_min)]
    create_network_data(
        seed_list=seed_list,
        network_num=network_num,
        lines=lines,
        max_minute=max_minute,
        list_depot=list_depot,
        d_time_table=d_time_table,
        list_recharge=list_recharge,
        nb_borne=nb_bornes,
        dict_length=dict_length,
        dict_length_hlp=dict_length_hlp,
        dict_lines=dict_lines,
        var_modif=var_modif
    )


# create_network_data(
#         seed_list=[9999],
#         network_num=network_num,
#         lines=lines,
#         max_minute=8,
#         list_depot=list_depot,
#         d_time_table=d_time_table,
#         list_recharge=list_recharge,
#         nb_borne=nb_bornes,
#         dict_length=dict_length,
#         dict_length_hlp=dict_length_hlp,
#         dict_lines=dict_lines,
#         var_modif=var_modif
#     )

