from datetime import datetime
import re

# va lire les fichiers en sortie et parse jusqua trouver le temps de resolution

# when at gerad : 
# when in local : testing_local/

def find_computing_time(line):

    nb = re.findall("\d+\.\d+", line)[0]
    return nb

with open('duals_inequalities_instances.txt', 'r') as f:
    instances_info_list = f.read().splitlines()

# when at gerad : ../MdevspGencolTest/
# when in local : testing_local/

computing_times = []

for instance_info in instances_info_list:
    def_info, ineq_info = instance_info.split('/')

    network, max_minute, seed = def_info.split('_')


    # 'reportProblem{}_default.out'.format(def_info)
    # 'reportProblem{}_{}.out'.format(def_info, ineq_info)

    with open('../MdevspGencolTest//reportProblem{}_default.out'.format(def_info)) as f:
        def_report = f.read().splitlines()
        computing_time = find_computing_time(def_report[-1])
        computing_times.append([network, max_minute, seed, 'default', computing_time])

        #computing_times[def_info]['default'] = float(find_computing_time(def_report[-1]))

    with open('../MdevspGencolTest//reportProblem{}_{}.out'.format(def_info, ineq_info)) as f:
        ineq_report = f.read().splitlines()
        computing_time = find_computing_time(ineq_report[-1])
        nb_inequalities = ineq_info.split('_')[1]
        computing_times.append([network, max_minute, seed, nb_inequalities, computing_time])

        #computing_times[def_info][ineq_info] = float(find_computing_time(ineq_report[-1]))

now = datetime.now()

dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")

with open('computing_times_{}.txt'.format(dt_string), 'w') as f:
    for c in computing_times:
        f.write(','.join(c))
        f.write('\n')


