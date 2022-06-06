from code_juliette.create_data_trip_shifting import create_network_data
from create_input_file import create_input
#from code_juliette.dataset_variables import d_time_table, dict_length, dict_length_hlp, dict_lines, lines, list_depot, list_recharge

dataset_num = '4a'
seed_list = [x for x in range(1)]
network_num = dataset_num+ '_0'
max_minute = 7 #1.3 permet d'avoir 1663 trajets et 6 permet d'avoir 385 trajets pour dataset4
#20 10 7 5 15 3 2 1.8 puis 
var_modif = 4 # j'ai utilisé que 1 pour l'instant
nb_borne = 2
nb_veh = 20
# 0: (7, 4, 2, 5) pour 1 dépot
# 1:(5,4,2,15) pour 2 depots
# 2: (2,4,5,35)
# 3: (1.4,4,5,50)

if dataset_num=='1a':
    from code_juliette.dataset_trip_shifting.dataset1_variables import d_time_table, dict_length, dict_length_hlp, dict_lines, lines, list_depot, list_recharge
elif dataset_num=='1b':
    from code_juliette.dataset_trip_shifting.dataset1b_variables import d_time_table, dict_length, dict_length_hlp, dict_lines, lines, list_depot, list_recharge
elif dataset_num=='2a':
    from code_juliette.dataset_trip_shifting.dataset2_variables import d_time_table, dict_length, dict_length_hlp, dict_lines, lines, list_depot, list_recharge
elif dataset_num=='2b':
    from code_juliette.dataset_trip_shifting.dataset2b_variables import d_time_table, dict_length, dict_length_hlp, dict_lines, lines, list_depot, list_recharge
elif dataset_num=='3a':
    from code_juliette.dataset_trip_shifting.dataset3_variables import d_time_table, dict_length, dict_length_hlp, dict_lines, lines, list_depot, list_recharge
elif dataset_num=='3b':
    from code_juliette.dataset_trip_shifting.dataset3b_variables import d_time_table, dict_length, dict_length_hlp, dict_lines, lines, list_depot, list_recharge
elif dataset_num=='4a':
    from code_juliette.dataset_trip_shifting.dataset4_variables import d_time_table, dict_length, dict_length_hlp, dict_lines, lines, list_depot, list_recharge
elif dataset_num=='4b':
    from code_juliette.dataset_trip_shifting.dataset4b_variables import d_time_table, dict_length, dict_length_hlp, dict_lines, lines, list_depot, list_recharge
elif dataset_num=='5a':
    from code_juliette.dataset_trip_shifting.dataset5_variables import d_time_table, dict_length, dict_length_hlp, dict_lines, lines, list_depot, list_recharge
elif dataset_num=='5b':
    from code_juliette.dataset_trip_shifting.dataset5b_variables import d_time_table, dict_length, dict_length_hlp, dict_lines, lines, list_depot, list_recharge

create_network_data(seed_list, network_num, lines, max_minute, list_depot, d_time_table, list_recharge, 
    nb_borne, dict_length, dict_length_hlp, dict_lines, var_modif=var_modif)
list_pb  = [str(network_num)+ "_" + str(x) for x in seed_list]
create_input(list_pb, nb_veh=nb_veh)