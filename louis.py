import argparse
import glob
import shutil
import string
from code_juliette.create_data_trip_shifting import create_network_data
from create_gencol_input import create_gencol_file
import os


# If default : 
# Read configurator file, create necessary networks and related instances

# If dual :
# Copy dualvariables files for each instances, and create related instances with dual inequalities


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

    # 1. Lit le fichier de configuration
    with open('configurator.txt', 'r') as f:
        configs = f.read().splitlines()

    default_pb_list_file = open('default_pb_list_file.txt', 'w')
    default_pb_list_file.close()

    # 2. Pour chaque config : on créer le network et le nb d'instances demandés
    for c in configs:
        nb_instances, network_num, max_minute, nb_borne, nb_veh = c.split(',')
        nb_instances = int(nb_instances)
        max_minute = int(max_minute) if float(max_minute).is_integer() else float(max_minute)
        nb_borne = int(nb_borne)
        nb_veh = int(nb_veh)

        create_network(nb_instances, network_num, max_minute, nb_borne)

        pb_list = [str(network_num) + "_" + str(max_minute).replace(".", "p") + "_" + str(i) for i in range(nb_instances)]

        for pb in pb_list:
            with open('default_pb_list_file.txt', 'a') as f:
                f.write('{}\n'.format(pb))
            
        create_gencol_file(pb_list, nb_veh=nb_veh, dual_variables_file_name='', nb_inequalities=0)

def create_duals_ineq_instances():

    # 1. Va lire les noms des problemes qu'on a résolu
    with open('default_pb_list_file.txt', 'r') as f:
        default_pb_list = f.read().splitlines()

    # 2. On va copier les variables duales de ces problèmes. 
    for pb in default_pb_list:
        shutil.copy('../MdevspGencolTest/dualVarsFirstLinearRelaxProblem{}_default.out'.format(pb), 'Networks/Network{}/'.format(pb))

    # 3. On lit le fichier config.
    with open('configurator.txt', 'r') as f:
        configs = f.read().splitlines()

    duals_inequalities_instances_file = open('duals_inequalities_instances.txt', 'w')
    duals_inequalities_instances_file.close()

    # 4. Pour chaque configuration, on re-creer un fichier gencol avec en entree le fichier de variables duales
    # On ajoute le nom de ce fichier (et le folder) dans un fichier texte en sortie
    for c in configs:
        nb_instances, network_num, max_minute, nb_borne, nb_veh = c.split(',')
        nb_instances = int(nb_instances)
        max_minute = int(max_minute) if float(max_minute).is_integer() else float(max_minute)
        nb_borne = int(nb_borne)
        nb_veh = int(nb_veh)

        for i in range(nb_instances):
            pb_name = '{}_{}_{}'.format(network_num, str(max_minute).replace('.', 'p'), i)  

            dual_variables_file_name = 'dualVarsFirstLinearRelaxProblem{}_default.out'.format(pb_name)

            dual_variables_file = open('Networks/Network{}/{}'.format(pb_name, dual_variables_file_name), 'r')

            nb_valid_dual_variables_values = 0

            dual_variables_list = dual_variables_file.read().splitlines()
            for line in dual_variables_list:

                dual_variable, value = line.split(' ')
                value = float(value)

                # Pour l'instant on laisse faire les +1000 (cout fixe) et les <0
                if value > 1 and value < 1000:
                    nb_valid_dual_variables_values += 1

            # 10%, 20%, 40%, 60%, 80%

            #for i in [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]:
            for i in [0.05, 0.1, 0.15, 0.25, 0.4, 0.5]:    

                nb_ineq = int(i*nb_valid_dual_variables_values)

                with open('duals_inequalities_instances.txt', 'a') as f:
                    f.write('{}/{}_{}\n'.format(pb_name,int(nb_ineq/nb_ineq), nb_ineq))

                # ici, ca devrait pas etre dual_Variables_file name, ca devrait etre le nom de input


                create_gencol_file([pb_name], nb_veh=nb_veh, dual_variables_file_name=dual_variables_file_name, nb_inequalities=nb_ineq, grp_size=nb_ineq)

def create_duals_ineq_instances_with_errors(experience_name):

    # 1. va chercher le inputProblem... dans gencol_files/{EXP}/...

    os.chdir('gencol_files/{}'.format(experience_name))
    
    for instance in glob.glob('*/'):

        instance_name = instance.replace('/', '')
        
        with open('{}/inputProblem{}_default.in'.format(instance_name, instance_name), 'r') as f:
            input_problem = f.read().splitlines()
            
            vehicule_i = 0

            for i, line in enumerate(input_problem):
                if "Columns" in line:
                    vehicule_i = i + 2
                    break

            nb_veh = int(input_problem[vehicule_i].split(' ')[4].replace(']', ''))

        print('{} : {}'.format(instance_name, nb_veh))

    os.chdir('../../')

    print(os.getcwd())

    for instance in glob.glob('Networks/{}/*/'.format(experience_name)):


        instance_name = instance.split('/')[2].replace('Network', '').replace('/', '')
        #print(instance_name)
        path_to_networks = 'Networks/{}'.format(experience_name)

        dual_variables_file_name = 'dualVarsFirstLinearRelaxProblem{}_default.out'.format(instance_name)
    
        # 5%, 15% et 25% de ABS(variable duale)
        # for percent in [0.05, 0.15, 0.25]:

        #     create_gencol_file([instance_name], path_to_networks=path_to_networks,nb_veh=nb_veh, dual_variables_file_name=dual_variables_file_name, take_absolute_value=True, percentage_ineq=percent, nb_grps=1)

        # 5%, 10%, 25% des inegalités
        # avec 5%, 10%, 25% d'erreurs
        for percent_ineq in [0.05, 0.1, 0.25, 0.5]:
            for percent_wrong in [0, 0.1, 0.25, 0.5, 0.75]:
                create_gencol_file([instance_name], path_to_networks=path_to_networks,nb_veh=nb_veh, dual_variables_file_name=dual_variables_file_name, take_absolute_value=False, percentage_ineq=percent_ineq, percentage_wrong=percent_wrong, nb_grps=1)

def create_duals_ineq_instances_pairwise(experience_name):
    # 1. va chercher le inputProblem... dans gencol_files/{EXP}/...

    os.chdir('gencol_files/{}'.format(experience_name))
    
    for instance in glob.glob('*/'):

        instance_name = instance.replace('/', '')
        
        with open('{}/inputProblem{}_default.in'.format(instance_name, instance_name), 'r') as f:
            input_problem = f.read().splitlines()
            
            vehicule_i = 0

            for i, line in enumerate(input_problem):
                if "Columns" in line:
                    vehicule_i = i + 2
                    break

            nb_veh = int(input_problem[vehicule_i].split(' ')[4].replace(']', ''))

        print('{} : {}'.format(instance_name, nb_veh))

    os.chdir('../../')

    print(os.getcwd())

    for instance in glob.glob('Networks/{}/*/'.format(experience_name)):


        instance_name = instance.split('/')[2].replace('Network', '').replace('/', '')
        #print(instance_name)
        path_to_networks = 'Networks/{}'.format(experience_name)

        dual_variables_file_name = 'dualVarsFirstLinearRelaxProblem{}_default.out'.format(instance_name)
    
        # 5%, 15% et 25% de ABS(variable duale)
        # for percent in [0.05, 0.15, 0.25]:

        #     create_gencol_file([instance_name], path_to_networks=path_to_networks,nb_veh=nb_veh, dual_variables_file_name=dual_variables_file_name, take_absolute_value=True, percentage_ineq=percent, nb_grps=1)

        # 5%, 10%, 25% des inegalités
        # avec 5%, 10%, 25% d'erreurs
        for percent_ineq in [0.05, 0.1, 0.25, 0.5]:
            create_gencol_file([instance_name], path_to_networks=path_to_networks,nb_veh=nb_veh, dual_variables_file_name=dual_variables_file_name, take_absolute_value=False, percentage_ineq=percent_ineq, nb_grps=1)

def create_duals_ineq_instantes_seq_grps(experience_name):

    # 1. va chercher le inputProblem... dans gencol_files/{EXP}/...

    os.chdir('gencol_files/{}'.format(experience_name))
    
    for instance in glob.glob('*/'):

        instance_name = instance.replace('/', '')
        
        with open('{}/inputProblem{}_default.in'.format(instance_name, instance_name), 'r') as f:
            input_problem = f.read().splitlines()
            
            vehicule_i = 0

            for i, line in enumerate(input_problem):
                if "Columns" in line:
                    vehicule_i = i + 2
                    break

            nb_veh = int(input_problem[vehicule_i].split(' ')[4].replace(']', ''))

        print('{} : {}'.format(instance_name, nb_veh))

    os.chdir('../../')

    print(os.getcwd())

    for instance in glob.glob('Networks/{}/*/'.format(experience_name)):


        instance_name = instance.split('/')[2].replace('Network', '').replace('/', '')
        #print(instance_name)
        path_to_networks = 'Networks/{}'.format(experience_name)

        dual_variables_file_name = 'dualVarsFirstLinearRelaxProblem{}_default.out'.format(instance_name)
    
        # 5%, 15% et 25% de ABS(variable duale)
        # for percent in [0.05, 0.15, 0.25]:

        #     create_gencol_file([instance_name], path_to_networks=path_to_networks,nb_veh=nb_veh, dual_variables_file_name=dual_variables_file_name, take_absolute_value=True, percentage_ineq=percent, nb_grps=1)

        # 5%, 10%, 25% des inegalités
        # avec 5%, 10%, 25% d'erreurs
        for percent_ineq in [0.05, 0.1, 0.25, 0.5]:
            for nb_grps in [2, 5, 10, 20]:
                create_gencol_file([instance_name], path_to_networks=path_to_networks,nb_veh=nb_veh, dual_variables_file_name=dual_variables_file_name, take_absolute_value=False, percentage_ineq=percent_ineq, nb_grps=nb_grps)
                
def create_one_dual_ineq(experience_name):
    # 1. va chercher le inputProblem... dans gencol_files/{EXP}/...

    os.chdir('gencol_files/{}'.format(experience_name))
    
    for instance in glob.glob('*/'):

        instance_name = instance.replace('/', '')
        
        with open('{}/inputProblem{}_default.in'.format(instance_name, instance_name), 'r') as f:
            input_problem = f.read().splitlines()
            
            vehicule_i = 0

            for i, line in enumerate(input_problem):
                if "Columns" in line:
                    vehicule_i = i + 2
                    break

            nb_veh = int(input_problem[vehicule_i].split(' ')[4].replace(']', ''))

        print('{} : {}'.format(instance_name, nb_veh))

    os.chdir('../../')

    print(os.getcwd())

    for instance in glob.glob('Networks/{}/*/'.format(experience_name)):


        instance_name = instance.split('/')[2].replace('Network', '').replace('/', '')
        #print(instance_name)
        path_to_networks = 'Networks/{}'.format(experience_name)

        dual_variables_file_name = 'dualVarsFirstLinearRelaxProblem{}_default.out'.format(instance_name)
    
        # 5%, 15% et 25% de ABS(variable duale)
        # for percent in [0.05, 0.15, 0.25]:

        #     create_gencol_file([instance_name], path_to_networks=path_to_networks,nb_veh=nb_veh, dual_variables_file_name=dual_variables_file_name, take_absolute_value=True, percentage_ineq=percent, nb_grps=1)

        # 5%, 10%, 25% des inegalités
        # avec 5%, 10%, 25% d'erreurs
        create_gencol_file([instance_name], path_to_networks=path_to_networks,nb_veh=nb_veh, dual_variables_file_name=dual_variables_file_name, take_absolute_value=False, percentage_ineq=0.01, percentage_wrong=0, nb_grps=1)

def create_all_ineq(experience_name):
    # 1. va chercher le inputProblem... dans gencol_files/{EXP}/...

    os.chdir('gencol_files/{}'.format(experience_name))
    
    for instance in glob.glob('*/'):

        instance_name = instance.replace('/', '')
        
        with open('{}/inputProblem{}_default.in'.format(instance_name, instance_name), 'r') as f:
            input_problem = f.read().splitlines()
            
            vehicule_i = 0

            for i, line in enumerate(input_problem):
                if "Columns" in line:
                    vehicule_i = i + 2
                    break

            nb_veh = int(input_problem[vehicule_i].split(' ')[4].replace(']', ''))

        print('{} : {}'.format(instance_name, nb_veh))

    os.chdir('../../')

    print(os.getcwd())

    for instance in glob.glob('Networks/{}/*/'.format(experience_name)):


        instance_name = instance.split('/')[2].replace('Network', '').replace('/', '')
        #print(instance_name)
        path_to_networks = 'Networks/{}'.format(experience_name)

        dual_variables_file_name = 'dualVarsFirstLinearRelaxProblem{}_default.out'.format(instance_name)
    
        # 5%, 15% et 25% de ABS(variable duale)
        # for percent in [0.05, 0.15, 0.25]:

        #     create_gencol_file([instance_name], path_to_networks=path_to_networks,nb_veh=nb_veh, dual_variables_file_name=dual_variables_file_name, take_absolute_value=True, percentage_ineq=percent, nb_grps=1)

        # 5%, 10%, 25% des inegalités
        # avec 5%, 10%, 25% d'erreurs

        create_gencol_file([instance_name], path_to_networks=path_to_networks,nb_veh=nb_veh, dual_variables_file_name=dual_variables_file_name, take_absolute_value=False, percentage_ineq=1, percentage_wrong=0, nb_grps=1)

def create_new_default(experience_name):

    # 1. va chercher le inputProblem... dans gencol_files/{EXP}/...

    os.chdir('gencol_files/{}'.format(experience_name))
    
    for instance in glob.glob('*/'):

        instance_name = instance.replace('/', '')
        
        with open('{}/inputProblem{}_default.in'.format(instance_name, instance_name), 'r') as f:
            input_problem = f.read().splitlines()
            
            vehicule_i = 0

            for i, line in enumerate(input_problem):
                if "Columns" in line:
                    vehicule_i = i + 2
                    break

            nb_veh = int(input_problem[vehicule_i].split(' ')[4].replace(']', ''))

        print('{} : {}'.format(instance_name, nb_veh))

    os.chdir('../../')

    print(os.getcwd())

    for instance in glob.glob('Networks/{}/*/'.format(experience_name)):


        instance_name = instance.split('/')[2].replace('Network', '').replace('/', '')
        #print(instance_name)
        path_to_networks = 'Networks/{}'.format(experience_name)

        #dual_variables_file_name = 'dualVarsFirstLinearRelaxProblem{}_default.out'.format(instance_name)
    
        # 5%, 15% et 25% de ABS(variable duale)
        # for percent in [0.05, 0.15, 0.25]:

        #     create_gencol_file([instance_name], path_to_networks=path_to_networks,nb_veh=nb_veh, dual_variables_file_name=dual_variables_file_name, take_absolute_value=True, percentage_ineq=percent, nb_grps=1)

        # 5%, 10%, 25% des inegalités
        # avec 5%, 10%, 25% d'erreurs

        create_gencol_file([instance_name], path_to_networks=path_to_networks,nb_veh=nb_veh, take_absolute_value=False, percentage_ineq=0, percentage_wrong=0, nb_grps=0, use_strong_task=False)

def create_new_grps_ineq(experience_name):

    # 1. va chercher le inputProblem... dans gencol_files/{EXP}/...

    os.chdir('gencol_files/{}'.format(experience_name))
    
    for instance in glob.glob('*/'):

        instance_name = instance.replace('/', '')
        
        with open('{}/inputProblem{}_default.in'.format(instance_name, instance_name), 'r') as f:
            input_problem = f.read().splitlines()
            
            vehicule_i = 0

            for i, line in enumerate(input_problem):
                if "Columns" in line:
                    vehicule_i = i + 2
                    break

            nb_veh = int(input_problem[vehicule_i].split(' ')[4].replace(']', ''))

        print('{} : {}'.format(instance_name, nb_veh))

    os.chdir('../../')

    print(os.getcwd())

    for instance in glob.glob('Networks/{}/*/'.format(experience_name)):


        instance_name = instance.split('/')[2].replace('Network', '').replace('/', '')
        #print(instance_name)
        path_to_networks = 'Networks/{}'.format(experience_name)

        dual_variables_file_name = 'dualVarsFirstLinearRelaxProblem{}_default.out'.format(instance_name)
    
        # 5%, 15% et 25% de ABS(variable duale)
        # for percent in [0.05, 0.15, 0.25]:

        #     create_gencol_file([instance_name], path_to_networks=path_to_networks,nb_veh=nb_veh, dual_variables_file_name=dual_variables_file_name, take_absolute_value=True, percentage_ineq=percent, nb_grps=1)

        # 5%, 10%, 25% des inegalités
        # avec 5%, 10%, 25% d'erreurs

        create_gencol_file([instance_name], path_to_networks=path_to_networks,nb_veh=nb_veh, dual_variables_file_name=dual_variables_file_name, take_absolute_value=False, percentage_ineq=0.5, percentage_wrong=0, nb_grps=1, use_strong_task=True, test_new_grp = True, new_grp_val_range=7)


parser = argparse.ArgumentParser()

parser.add_argument('type', type=str)
parser.add_argument('experience_name', nargs='?', default='')

args = parser.parse_args()

if args.type == 'default':
    create_defaults_instances()
elif args.type == 'dual':
    create_duals_ineq_instances()
elif args.type == 'errors':
    create_duals_ineq_instances_with_errors(args.experience_name)
elif args.type == 'pairwise':
    create_duals_ineq_instances_pairwise(args.experience_name)
elif args.type == 'seq_grps':
    create_duals_ineq_instantes_seq_grps(args.experience_name)
elif args.type == 'one':
    create_one_dual_ineq(args.experience_name)
elif args.type == 'all':
    create_all_ineq(args.experience_name)
elif args.type == 'new_def':
    create_new_default(args.experience_name)
elif args.type == 'new_grps':
    create_new_grps_ineq(args.experience_name)
else:
    print('wrong args')

