# Sert a resoudre un type dinstances

import sys
import os
import shutil
import subprocess
import time

# 1. copier toutes les instances à run dans le folder MDEVSP

# 2. executer gencol

def solve_instances(list_instances):

    nb_cpu = 4
    out_file = open("out", "w")
    out_file.close()

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


def copy_files(suffix): 

    files_to_run = []

    suffix = sys.argv[1]

    for folder in os.listdir('Networks'):

        if 'Network' not in folder:
            continue

        for file in os.listdir('Networks/{}'.format(folder)):

            if suffix in file:

                shutil.copy('Networks/{}/{}'.format(folder, file), '../MdevspGencolTest/')

                pb_name = file.replace('inputProblem', '').replace('.in', '')

                files_to_run.append(pb_name)

                continue
    return files_to_run

# change le working directtory pour execute

# os.chdir('/home/popoloui/MdevspGencol/MdevspGencolTest')

# solve_instances(instances_to_execute)

if __name__ == '__main__':


    if len(sys.argv) <= 1:
        print('Missing suffix')
        sys.exit()

    suffix = sys.argv[1]

    files_to_run = copy_files(suffix=suffix)

    print(files_to_run)

    os.chdir('/home/popoloui/MdevspGencol/MdevspGencolTest')

    solve_instances(files_to_run)
