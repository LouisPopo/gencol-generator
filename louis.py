import argparse
import shutil
import string
from code_juliette.create_data_trip_shifting import create_network_data
from create_gencol_input import create_gencol_file

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

def create_defaults_instances():

    # va lire un fichier .txt et génère les networks ainsi que les fichiers d'inputs gencol correspondants
    with open('configurator.txt', 'r') as f:
        configs = f.read().splitlines()

    default_pb_list_file = open('default_pb_list_file.txt', 'w')

    for c in configs:
        nb_instances, network_num, max_minute, nb_borne, nb_veh = c.split(',')
        nb_instances = int(nb_instances)
        max_minute = int(max_minute) if float(max_minute).is_integer() else float(max_minute)
        nb_borne = int(nb_borne)
        nb_veh = int(nb_veh)

        create_network(nb_instances, network_num, max_minute, nb_borne)

        pb_list = [str(network_num) + "_" + str(max_minute).replace(".", "p") + "_" + str(i) for i in range(nb_instances)]

        for pb in pb_list:
            default_pb_list_file.write('{}\n'.format(pb))
        
        create_gencol_file(pb_list, nb_veh=nb_veh, dual_variables_file_name='', nb_inequalities=0)

def create_duals_ineq_instances():
    # va chercher les fichiers de variables duales, 
    # les copies dans le folder network correspondant
    # run le genereate(avec fichier de dual variables)
    with open('default_pb_list_file.txt', 'r') as f:
        default_pb_list = f.read().splitlines()

    for pb in default_pb_list:

        print(pb)

        shutil.copy('/MdevspGencolTest/dualVarsFirstLinearRelaxProblem{}.out'.format(pb), 'Networks/Network{}/'.format(pb))

        print('From : /MdevspGencolTest/dualVarsFirstLinearRelaxProblem{}.out'.format(pb))
        print('To : Networks/Network{}/'.format(pb))



parser = argparse.ArgumentParser()

parser.add_argument('type', type=str)

args = parser.parse_args()

if args.type == 'default':
    create_defaults_instances()
elif args.type == 'dual':
    create_duals_ineq_instances
    #print('allp')
else:
    print('wrong args')

