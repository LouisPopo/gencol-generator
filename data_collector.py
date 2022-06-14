from datetime import datetime
import os, glob
import re
import argparse

# va lire les fichiers en sortie et parse jusqua trouver le temps de resolution

# when at gerad : 
# when in local : testing_local/

def find_computing_time(line):

    nb = re.findall("\d+\.\d+", line)[0]
    return nb

def find_col_gen_it(report):
    for line in report:
        if "Column generation iterations" in line:

            for s in line.split():
                if s.isdigit():
                    return s

def default_aggregation():

    with open('duals_inequalities_instances.txt', 'r') as f:
        instances_info_list = f.read().splitlines()

    # when at gerad : ../MdevspGencolTest/
    # when in local : testing_local/

    computing_times = []
    columns_generations_it = []

    logged_defaults = set()

    print(instances_info_list)

    for instance_info in instances_info_list:
        def_info, ineq_info = instance_info.split('/')

        network, max_minute, seed = def_info.split('_')


        # 'reportProblem{}_default.out'.format(def_info)
        # 'reportProblem{}_{}.out'.format(def_info, ineq_info)

        if def_info not in logged_defaults and os.path.exists('../MdevspGencolTest/reportProblem{}_default.out'.format(def_info)):
            
            logged_defaults.add(def_info)

            with open('../MdevspGencolTest/reportProblem{}_default.out'.format(def_info)) as f:
                def_report = f.read().splitlines()
                computing_time = find_computing_time(def_report[-1])
                computing_times.append([network, max_minute, seed, 'default', computing_time])

                col_gen_it = find_col_gen_it(def_report)
                columns_generations_it.append([network, max_minute, seed, 'default', col_gen_it])

                #computing_times[def_info]['default'] = float(find_computing_time(def_report[-1]))

        if os.path.exists('../MdevspGencolTest/reportProblem{}_{}.out'.format(def_info, ineq_info)):
            with open('../MdevspGencolTest/reportProblem{}_{}.out'.format(def_info, ineq_info)) as f:
                ineq_report = f.read().splitlines()
                computing_time = find_computing_time(ineq_report[-1])
                nb_inequalities = ineq_info.split('_')[1]
                computing_times.append([network, max_minute, seed, nb_inequalities, computing_time])

                col_gen_it = find_col_gen_it(ineq_report)
                columns_generations_it.append([network, max_minute, seed, nb_inequalities, col_gen_it])

                #computing_times[def_info][ineq_info] = float(find_computing_time(ineq_report[-1]))

    now = datetime.now()

    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")

    with open('computing_times_{}.txt'.format(dt_string), 'w') as f:
        for c in computing_times:
            f.write(','.join(c))
            f.write('\n')

    with open('columns_generations_{}.txt'.format(dt_string), 'w') as f:
        for c in columns_generations_it:
            f.write(','.join(c))
            f.write('\n')


def folder_aggregation(folder_path):

    os.chdir(folder_path)
    for file in glob.glob('reportProblem*'):
        network, mm, seed, nb_grps, grps_size = file.replace('reportProblem', '').replace('.out', '').split('_')
        print('{} {} {} {} {}'.format(network, mm, seed, nb_grps, grps_size))

    print(folder_path)

parser = argparse.ArgumentParser()

parser.add_argument('folder_path', type=str, nargs='?', default='')

args = parser.parse_args()

if args.folder_path == '':
    default_aggregation()
elif args.folder_path != '':
    folder_aggregation(args.folder_path)
else:
    print('wrong args')