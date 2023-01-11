# Sert a resoudre un type dinstances

import sys
import os
import shutil
import subprocess
import time

# 1. copier toutes les instances à run dans le folder MDEVSP

# 2. executer gencol

def solve_instances(list_instances, cpu_int):

    nb_cpu = 2

    out_file_name = "out_{}".format(cpu_int)
    out_file = open(out_file_name, "w")
    out_file.close()

    list_process = []
    idx_active = []

    pb_names = [f.split('/')[-1].replace('inputProblem', '').replace('.in', '') for f in list_instances]

    for i, pb in enumerate(pb_names) :
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
                    with open(out_file_name, 'a') as out:
                        out.write(f'Done Problem{list_instances[j]} ({len(list_process)-len(idx_active)}/{len(list_instances)})\n')

    while len(idx_active)>0:
        time.sleep(0.1)
        for j in idx_active:
            done = list_process[j].poll()
            if done is not None:
                idx_active.remove(j)
                with open(out_file_name, 'a') as out:
                    out.write(f'Done Problem{list_instances[j]} ({len(list_process)-len(idx_active)}/{len(list_instances)})\n') 


def copy_files(suffix, min_seed, cpu_int, nb_cpus): 

    files_to_run = []

    for folder in os.listdir('Networks'):

        if 'Network' not in folder:
            continue

        instance_seed = int(folder.split('_')[-1])
        if instance_seed < min_seed:
            continue
    #

        for file in os.listdir('Networks/{}'.format(folder)):

            if suffix in file and 'inputProblem' in file:

                files_to_run.append('Networks/{}/{}'.format(folder, file))

                continue
    
    # files_to_run = sorted(files_to_run, key= lambda f: (
    #     f.split('/')[1].split('_')[1], 
    #     int(f.split('/')[1].split('_')[2])))
    files_to_run_on_cpu = files_to_run[(cpu_int - 1)::nb_cpus]

    for f in files_to_run_on_cpu:
        shutil.copy(f, '../MdevspGencolTest/')

    return files_to_run_on_cpu

# change le working directtory pour execute

# os.chdir('/home/popoloui/MdevspGencol/MdevspGencolTest')

# solve_instances(instances_to_execute)

if __name__ == '__main__':


    if len(sys.argv) <= 2:
        print('Missing arguments : suffix, min_seed, cpu_int, nb_cpus')
        sys.exit()

    suffix = sys.argv[1]
    min_seed = int(sys.argv[2])
    cpu_int = int(sys.argv[3])
    nb_cpus = int(sys.argv[4])

    files_to_run = copy_files(suffix=suffix, min_seed=min_seed, cpu_int=cpu_int, nb_cpus=nb_cpus)

    print(len(files_to_run))
    print(files_to_run)

    os.chdir('/home/popoloui/MdevspGencol/MdevspGencolTest')

    solve_instances(files_to_run, cpu_int = cpu_int)
