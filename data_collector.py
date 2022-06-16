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

def find_obj_value(report):
    for line in report:
        if "Best relaxation cost" in line:

            nb = re.findall("\d+\.\d+", line)[0]

            return nb

def default_aggregation_wrong_ineq():

    os.chdir('../MdevspGencolTest')

    data = []

    for file in glob.glob('report*'):
        network, mm, seed, nb_grp, nb_ineq, w, nb_wrong = file.replace('reportProblem', '').replace('.out', '').split('_')

        with open(file, 'r') as f:
            r = f.read().splitlines()

            computing_time = find_computing_time(r[-1])
            col_gen_it = find_col_gen_it(r)
            obj_val = find_obj_value(r)

            data.append([network, mm, seed, nb_ineq, nb_wrong, computing_time, col_gen_it, obj_val])
    
    data.sort(key = lambda x : (x[2], x[4], x[6])) # seed, nb_ineq, nb_wrong

    now = datetime.now()

    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")

    with open('infos_{}.txt'.format(dt_string), 'w') as f:
        for c in data:
            f.write(','.join(c))
            f.write('\n')


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

    obj_values = []

    os.chdir(folder_path)
    for file in glob.glob('reportProblem*'):

        if "default" in file:
            network, mm, seed, grps_size = file.replace('reportProblem', '').replace('.out', '').split('_')
            nb_grps = '1'
        else:
            network, mm, seed, nb_grps, grps_size = file.replace('reportProblem', '').replace('.out', '').split('_')
        
        with open(file) as f:
            report = f.read().splitlines()

            obj_value = find_obj_value(report)

            obj_values.append([network, mm, seed, nb_grps, grps_size, obj_value])

    now = datetime.now()

    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")

    obj_values.sort(key= lambda x: (x[2], x[4]))

    with open('objective_values_{}.txt'.format(dt_string), "w") as f:
        for c in obj_values:
            f.write(','.join(c))
            f.write('\n')
    

    print(folder_path)

parser = argparse.ArgumentParser()

parser.add_argument('ineq', type=str)
parser.add_argument('folder_path', type=str, nargs='?', default='')

args = parser.parse_args()

if args.ineq == 'wrong':
    default_aggregation_wrong_ineq()
elif args.folder_path == '':
    default_aggregation()
elif args.folder_path != '':
    folder_aggregation(args.folder_path)
else:
    print('wrong args')