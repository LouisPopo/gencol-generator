import argparse
import shutil

from create_gencol_input import create_gencol_file

parser = argparse.ArgumentParser()

parser.add_argument('pb', type=str)
parser.add_argument('nb_veh', type=str)
parser.add_argument('nb_ineq', type=str)

args = parser.parse_args()

problem = args.pb # '4b_1p4_7'

# print(args.pb)
# print(args.nb_veh)
# print(args.nb_bornes)   

dual_variables_file_name = 'dualVarsFirstLinearRelaxProblem{}_default.out'.format(problem)

# ../MdevspGencolTest

shutil.copy('../MdevspGencolTest/dualVarsFirstLinearRelaxProblem{}_default.out'.format(problem), 'Networks/Network{}/'.format(problem))

create_gencol_file([problem], nb_veh=int(args.nb_veh), dual_variables_file_name=dual_variables_file_name, nb_inequalities=int(args.nb_ineq), grp_size=int(args.nb_ineq), random_ineq=True)


# duals_inequalities_instances_file = open('duals_inequalities_instances.txt', 'w')
# duals_inequalities_instances_file.close()

# # 4. Pour chaque configuration, on re-creer un fichier gencol avec en entree le fichier de variables duales
# # On ajoute le nom de ce fichier (et le folder) dans un fichier texte en sortie
# for c in configs:
#     nb_instances, network_num, max_minute, nb_borne, nb_veh = c.split(',')
#     nb_instances = int(nb_instances)
#     max_minute = int(max_minute) if float(max_minute).is_integer() else float(max_minute)
#     nb_borne = int(nb_borne)
#     nb_veh = int(nb_veh)

#     for i in range(nb_instances):
#         pb_name = '{}_{}_{}'.format(network_num, str(max_minute).replace('.', 'p'), i)  

#         dual_variables_file_name = 'dualVarsFirstLinearRelaxProblem{}_default.out'.format(pb_name)

#         dual_variables_file = open('Networks/Network{}/{}'.format(pb_name, dual_variables_file_name), 'r')

#         nb_valid_dual_variables_values = 0

#         dual_variables_list = dual_variables_file.read().splitlines()
#         for line in dual_variables_list:

#             dual_variable, value = line.split(' ')
#             value = float(value)

#             # Pour l'instant on laisse faire les +1000 (cout fixe) et les <0
#             if value > 1 and value < 1000:
#                 nb_valid_dual_variables_values += 1

#         # 10%, 20%, 40%, 60%, 80%

#         for i in [0.1, 0.2, 0.4, 0.6, 0.8]:

#             nb_ineq = int(i*nb_valid_dual_variables_values)

#             with open('duals_inequalities_instances.txt', 'a') as f:
#                 f.write('{}/{}_{}\n'.format(pb_name,int(nb_ineq/nb_ineq), nb_ineq))

#             # ici, ca devrait pas etre dual_Variables_file name, ca devrait etre le nom de input


#             create_gencol_file([pb_name], nb_veh=nb_veh, dual_variables_file_name=dual_variables_file_name, nb_inequalities=nb_ineq, grp_size=nb_ineq)