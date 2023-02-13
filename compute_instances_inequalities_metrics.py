import re
import pandas as pd
from glob import glob
import sys

pd.options.mode.chained_assignment = None  # default='warn'
# this file computes instances metrics:
# model accuracy in general (ALL predictions made)
# the accuracy if we have a treshold of 0.2, 0.25, 0.3, 0.35

def compute(min_id, suffix):

    results = []

    for instance_folder in glob('Networks/Network*'):

        instance = instance_folder.replace('Networks/Network','')

        instance_id = int(instance.split('_')[-1])

        if instance_id < min_id:
            continue
        
        input_file_name = None
        dual_vars_file_name = None
        for file in glob('Networks/Network{}/*'.format(instance)):

            if 'input' in file and suffix in file:
                input_file_name = file
            elif 'default' in file and 'dualVars' in file:
                dual_vars_file_name = file

        if input_file_name is None or dual_vars_file_name is None:
            print('{} is missing'.format(instance))
            continue


        # 1. Read input file to get series of ineq
        with open(input_file_name) as f:
            ineqs = []
            for ln in f:
                if ln.startswith('Y_'):
                    covers = re.findall(r"\b\w*Cover\w*\b", ln)
                    ineqs.append(covers)           
        
        
        series = []
        current_serie = []
        
        for i in range(len(ineqs) -1):
            if len(current_serie) == 0:
                current_serie.append(ineqs[i][0])
            current_serie.append(ineqs[i][1])
            if ineqs[i][1] != ineqs[i+1][0]:
                series.append(current_serie.copy())
                current_serie = []
        current_serie.append(ineqs[-1][1])
        series.append(current_serie.copy())

        # 2. Read dual vars to get values

        pi_vals = pd.read_csv(dual_vars_file_name, sep=' ', names=['name', 'pi_value'])
        pi_vals = pi_vals[pi_vals['name'].str.startswith('Cover')]
        pi_vals = dict(zip(pi_vals['name'], pi_vals['pi_value']))
        
        # 3. Compute metrics
        
        # 3.1. Metric : overall sortedness
        # pour chaque serie, pour chaque paire possible, on regarde si l'inégalitée est bonne
        # i >= i + 1 >= i + 2, etc..
        nb_pairs = 0
        nb_pair_error = 0
        for serie in series:
            for i in range(len(serie) - 1):
                i_val = pi_vals[serie[i]]
                for j in range(i+1, len(serie)):
                    j_val = pi_vals[serie[j]]

                    if i_val < j_val:
                        nb_pair_error += 1
                    
                    nb_pairs += 1
        overall_sortedness = 1 - (nb_pair_error/nb_pairs)
        # print('{}/{} wrong ({})'.format(nb_pair_error, nb_pairs, nb_pair_error/nb_pairs))

        # 3.2. For each ineq added, check if real
        nb_wrong_ineq = 0
        for serie in series:
            
            for i in range(len(serie) - 1):
                i_val = pi_vals[serie[i]]
                j_val = pi_vals[serie[i+1]]

                if i_val < j_val:
                    nb_wrong_ineq += 1

        nb_ineq = sum([len(s) - 1 for s in series])
        nb_good_ineq = nb_ineq - nb_wrong_ineq
        percent_good = nb_good_ineq / nb_ineq 

        net, max_min, seed = instance.split('_')
        results.append([net, max_min, seed, nb_ineq, nb_good_ineq, nb_wrong_ineq, percent_good, overall_sortedness])

        print('Done for : {}'.format(instance))
    
    df_results = pd.DataFrame(results, columns=[
        'net',
        'max_min',
        'seed',
        'nb ineq', 
        'nb good ineq',
        'nb wrong ineq',
        'percent good ineq',
        'overall sortedness'
    ])

    df_results.sort_values(['max_min', 'seed'], inplace=True)

    df_results.to_csv('instances_inequalities_metrics_{}.csv'.format(suffix), index=False)


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print('Missing arguments : suffix or output_name')
        sys.exit()
    
    suffix = sys.argv[1]

    compute(175, suffix)