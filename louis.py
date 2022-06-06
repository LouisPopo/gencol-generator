from code_juliette.create_data_trip_shifting import create_network_data
from create_gencol_input import create_gencol_file


# nb_instances = 10   # Nombre d'instances 
# dataset_num = '4b'
# seed_list = [x for x in range(nb_instances)]
# max_minute = 25 #1.3 permet d'avoir 1663 trajets et 6 permet d'avoir 385 trajets pour dataset4
#20 10 7 5 15 3 2 1.8 puis ma
# str_max = str(max_minute).replace('.', 'p')
# network_num = dataset_num
# var_modif = 1 # j'ai utilisé que 1 pour l'instant
# nb_borne = 4
# nb_veh = 20

def create_network(nb_instances, dataset_num, max_minute, nb_borne):

    seed_list = [x for x in range(nb_instances)]
    network_num = dataset_num

    var_modif = 1

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

# fonction qui génère les instances par défaut (sans inégalités duales)
def create_defaults_instances():

    # va lire un fichier .txt et génère les networks ainsi que les fichiers d'inputs gencol correspondants
    with open('configurator.txt', 'r') as f:
        configs = f.read().splitlines()

    for c in configs:
        print(c)
        nb_instances, network_num, max_minute, nb_borne, nb_veh = c.split(',')
        nb_instances = int(nb_instances)
        max_minute = int(max_minute) if float(max_minute).is_integer() else float(max_minute)
        nb_borne = int(nb_borne)
        nb_veh = int(nb_veh)

        create_network(nb_instances, network_num, max_minute, nb_borne)

        pb_list = [str(network_num) + "_" + str(max_minute).replace(".", "p") + "_" + str(i) for i in range(nb_instances)]

        print(pb_list)
        
        create_gencol_file(pb_list, nb_veh=nb_veh, dual_variables_file_name='', nb_inequalities=0)
    # 1. should check if network exists

    # pb_list = [str(network_num)+ "_" + str(x) for x in seed_list]

    # 
create_defaults_instances()

# On veut creer des inegalites duales

# for i in range(3):
#     for nb_in in [100,200,300]:
#         for grp_size in [50,100,200,300]:
#             if grp_size > nb_in:
#                 continue
#             create_gencol_file(pb_list, nb_veh=nb_veh, dual_variables_file_name='dualVarsFirstLinearRelaxProblem4b_0_604_0.out', nb_inequalities=nb_in, grp_size=grp_size, id=i)

# # 1st set of tests
# # for i in range(3):
# #     for t in [True, False]:
# #         for n in [1, 5, 10, 25, 50, 100, 200, 275]:
# #             create_gencol_file(pb_list, nb_veh=nb_veh, dual_variables_file_name='dualVarsFirstLinearRelaxProblem5a_0_460_0.out', nb_inequalities = n, sequencial_inequalities=t, id=i)

# # 2nd set of tests



# for nb_in in [100,200]:
#     for grp_size in [10,25,50,100]:
#         create_gencol_file(pb_list, nb_veh=nb_veh, dual_variables_file_name='dualVarsFirstLinearRelaxProblem5a_0_460_0.out', nb_inequalities=nb_in, grp_size=grp_size)
