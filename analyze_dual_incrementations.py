
from glob import glob
import pandas as pd
import re

def get_bound_increments_infos(suffix, min_id):

    results = []

    nb_instances = 0

    for instance_folder in glob('Networks/Network*'):
        
        instance = instance_folder.replace('Networks/Network','')

        instance_id = int(instance.split('_')[-1])

        if instance_id < min_id:
            continue

        net, max_min, seed = instance.split('_')

        def_report_file = None
        inst_out_file = None
        dual_vars_file_name = None
        inst_input_file = None
        for f in glob('{}/*'.format(instance_folder)):

            if 'report' in f and 'default' in f:
                def_report_file = f
            elif 'out' in f and suffix in f and '.' not in f:
                inst_out_file = f
            elif 'default' in f and 'dualVars' in f:
                dual_vars_file_name = f
            elif 'inputProblem' in f and suffix in f:
                inst_input_file = f

        if def_report_file is None or inst_out_file is None or dual_vars_file_name is None or inst_input_file is None:
            print('{} is missing a file'.format(instance_folder))
            continue

        
        # 1. Read input file to get series of ineq
        with open(inst_input_file) as f:
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
        # pour avoir les series

        pi_vals = pd.read_csv(dual_vars_file_name, sep=' ', names=['name', 'pi_value'])
        pi_vals = pi_vals[pi_vals['name'].str.startswith('Cover')]
        pi_vals = dict(zip(pi_vals['name'], pi_vals['pi_value']))

        var_dict = dict()

        with open(inst_input_file, 'r') as f:

            for ln in f.readlines():
                if 'Y_' in ln:
                    var = re.findall('Y_[0-9]+', ln)[0]
                    covers = re.findall('Cover_T[0-9]+', ln)
                    var_dict[var] = covers

        with open(inst_out_file, 'r') as f:
            
            itr = 0
            for ln in f.readlines():

                if "Stop solving model `(default)':" in ln:
                    itr += 1
                    continue
                elif 'dual inequality' in ln:
                    var = re.findall('Y_[0-9]+', ln)[0]
                    val = float(re.findall('[0-9]\.[0-9]+', ln)[0])

                    pi_a, pi_b = var_dict[var]
                    pi_a_val = pi_vals[pi_a]
                    pi_b_val = pi_vals[pi_b]

                    good_ineq = 1 if pi_a_val >= pi_b_val else 0

                    results.append([net, max_min, seed, itr, var, val, pi_a, pi_a_val, pi_b, pi_b_val, good_ineq])
                    continue
                elif 'Linear relaxation solution' in ln:
                    break

        print('Done with : {}'.format(instance_folder))

    df_results = pd.DataFrame(results, columns=[
        'net',
        'max_min',
        'seed',
        'iter',
        'variable',
        'dual_value',
        'pi_a',
        'pi_a val',
        'pi_b',
        'pi_b val',
        'good'
    ])

    df_results.sort_values(['max_min', 'seed', 'iter', 'dual_value'], ascending=[True, True, True, False], inplace=True)

    df_results.to_csv('dual_values_analysis_{}.csv'.format(suffix), index=False)
                
                    

if __name__ == '__main__':
    get_bound_increments_infos('ST_COEFNaN_RANGENaN_TRESH0p25', 175)