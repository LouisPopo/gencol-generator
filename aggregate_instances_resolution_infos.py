# va chercher les infos de toutes les instances résolues et produit un csv :

# parametres, nombre de trajets, temps de resolution, val. sol. optimale, nombre de bus utilisés, etc. 


from glob import glob
import os
import shutil

import re
for instance_folder in glob('Networks/Network*'):

    instance_info = instance_folder.split('/')[1].replace('Network', '')

    dual_vars_path = '{}/dualVarsFirstLinearRelaxProblem{}_default.out'.format(instance_folder, instance_info)
    report_file_path = '{}/reportProblem{}_default.out'.format(instance_folder, instance_info)

    if os.path.exists(report_file_path):

        with open(report_file_path, 'r') as f:

            txt_lines = f.readlines()
            
            time_line = txt_lines[3]

            time = float(time_line[time_line.find("(")+1:time_line.find(")")].split(' ')[-1])
            
            obj_val_line = txt_lines[6]

            obj_val = float(obj_val_line.split(':')[1].split(' ')[-1])

            print('{} : {}'.format(time, obj_val))

