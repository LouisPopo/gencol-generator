import argparse
import os
import shutil
import subprocess

import time

from numpy import insert
nb_cpu = 2
out_file = open("out", "w")
out_file.close()

def solve_instances(list_instances):
    list_process = []
    idx_active = []
    for i, pb in enumerate(list_instances) :
        #cpu_free -= 1
        idx_active.append(i)
        with open('out'+pb, 'w') as f:
            list_process.append(subprocess.Popen(['ExecMdevspGencol','Problem'+pb], stdout=f, stderr=f))
        while len(idx_active)>=nb_cpu:
            time.sleep(0.1)
            for j in idx_active:
                done = list_process[j].poll()
                if done is not None:
                    idx_active.remove(j)
                    #cpu_free += 1
                    with open('out', 'a') as out:
                        out.write(f'Done Problem{list_instances[j]} ({len(list_process)-len(idx_active)}/{len(list_instances)})\n')

    while len(idx_active)>0:
        time.sleep(0.1)
        for j in idx_active:
            done = list_process[j].poll()
            if done is not None:
                idx_active.remove(j)
                with open('out', 'a') as out:
                    out.write(f'Done Problem{list_instances[j]} ({len(list_process)-len(idx_active)}/{len(list_instances)})\n') 


# 1. va runner les instances par default
# Pour que ca marche, les input doivent etre copies dans le folder ...GencolTest


def default_resolution():

    with open('default_pb_list_file.txt', 'r') as f:
        default_pb_list = f.read().splitlines()

    instances_to_execute = []

    for pb in default_pb_list:

        # Copy le input file a partir du folder ou il est genere et ou gencol s'execute
        shutil.copy('gencol_files/{}/inputProblem{}_default.in'.format(pb, pb), '../MdevspGencolTest/')

        # Rajoute le nom pour Exec
        instances_to_execute.append('{}_default'.format(pb))

    # change le working directtory pour execute
    os.chdir('/home/popoloui/MdevspGencol/MdevspGencolTest')

    solve_instances(instances_to_execute)

def dual_ineq_resolution():

    # va ouvrir le fichier des problemes

    with open('duals_inequalities_instances.txt', 'r') as f:
        duals_ineq_path_info = f.read().splitlines()

    instances_to_execute = []

    for instance_path_info in duals_ineq_path_info:

        def_info, ineq_info = instance_path_info.split('/')

        shutil.copy('gencol_files/{}/inputProblem{}_{}.in'.format(def_info, def_info, ineq_info), '../MdevspGencolTest/')

        # Rajoute le nom pour Exec
        # instance_name = instance_path.split('/')[1].replace('inputProblem', '')
        instances_to_execute.append('{}_{}'.format(def_info, ineq_info))

    # change le working directtory pour execute
    os.chdir('/home/popoloui/MdevspGencol/MdevspGencolTest')


    print(instances_to_execute)

    solve_instances(instances_to_execute)


parser = argparse.ArgumentParser()

parser.add_argument('type', type=str)

args = parser.parse_args()

if args.type == 'default':
    default_resolution()
elif args.type == 'dual':
    dual_ineq_resolution()
else:
    print('wrong args')



        

    

