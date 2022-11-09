# va chercher les infos de toutes les instances résolues et produit un csv :

# parametres, nombre de trajets, temps de resolution, val. sol. optimale, nombre de bus utilisés, etc. 

from glob import glob
import os
import shutil
import pandas as pd

results = []
for instance_folder in glob('Networks/Network*'):

    instance_info = instance_folder.split('/')[1].replace('Network', '')

    network, max_min, seed = instance_info.split('_')

    dual_vars_file_path = '{}/dualVarsFirstLinearRelaxProblem{}_default.out'.format(instance_folder, instance_info)
    report_file_path = '{}/reportProblem{}_default.out'.format(instance_folder, instance_info)
    out_file_path = '{}/out{}_default'.format(instance_folder, instance_info)
    trips_file_path = '{}/voyages.txt'.format(instance_folder)

    if os.path.exists(report_file_path):

        with open(report_file_path, 'r') as f:

            txt_lines = f.readlines()
            
            time_line = txt_lines[3]

            time = float(time_line[time_line.find("(")+1:time_line.find(")")].split(' ')[-1])
            
            obj_val_line = txt_lines[6]

            obj_val = float(obj_val_line.split(':')[1].split(' ')[-1])

            col_gen_iterations_line = txt_lines[21]

            col_gen_iterations = int(col_gen_iterations_line.split(':')[1].strip())

            mem_use_line = txt_lines[55]

            tot_mem_use = float(mem_use_line.split(':')[1].strip())

        with open(out_file_path, 'r') as f:

            gotV0, gotV1 = False, False

            for line in f:

                if 'Veh_D0' in line:

                    nb_Veh_D0 = float(line.split(' ')[4])

                    gotV0 = True
                
                elif 'Veh_D1' in line:

                    nb_Veh_D1 = float(line.split(' ')[4])

                    gotV1 = True

                if gotV0 and gotV1 :
                    break

        with open(trips_file_path, 'r') as f:

            nb_trips = len(f.readlines())

        results.append([network, max_min, int(seed), nb_trips, nb_Veh_D0, nb_Veh_D1, obj_val, time, col_gen_iterations, tot_mem_use])
    

df_results = pd.DataFrame(results, columns=['net', 'max_min', 'seed', 'nb_trips', 'nb_veh_D0', 'nb_veh_D1', 'obj_val', 'sol_time', 'col_gen_iter', 'tot_mem_use']) 

df_results.sort_values(['max_min', 'seed'], inplace=True)

df_results.to_csv('instances_stats.csv', index=False)       
    
