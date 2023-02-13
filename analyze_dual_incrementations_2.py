
from glob import glob
import pandas as pd
import re

suffix = 'ST_COEFNaN_RANGENaN_TRESH0p25'

for instance_folder in glob('Networks/Network*'):
        
    instance = instance_folder.replace('Networks/Network','')

    instance_id = int(instance.split('_')[-1])

    if instance_id < 175:
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

    pi_vals = pd.read_csv(dual_vars_file_name, sep=' ', names=['name', 'pi_value'])
    pi_vals = pi_vals[pi_vals['name'].str.startswith('Cover')]
    pi_vals = dict(zip(pi_vals['name'], pi_vals['pi_value']))

    print(pi_vals)

    dual_vars = dict()

    # 1. Read input file to get series of ineq
    nb_serie = 0
    pos_serie = 0
    previous_B = None
    with open(inst_input_file) as f:
        for ln in f:
            if ln.startswith('Y_'):
                y_var = ln.split(' ')[0]
                covers = re.findall(r"\b\w*Cover\w*\b", ln)
                
                pi_a_val = pi_vals[covers[0]]
                pi_b_val = pi_vals[covers[1]]
                good = (pi_a_val >= pi_b_val)

                if covers[0] != previous_B :
                    # nouvelle serie:
                    nb_serie += 1
                    pos_serie = 0
                else:
                    pos_serie += 1

                previous_B = covers[1]
                
                dual_vars[y_var] = {
                    'serie' : nb_serie,
                    'pos_serie' : pos_serie,
                    'dual_var_value' : 0,
                    'good' : good,
                    'pi_a' : covers[0],
                    'pi_a_val' : pi_a_val,
                    'pi_b' : covers[1],
                    'pi_b_val' : pi_b_val
                }

    with open(inst_out_file, 'r') as f:
            
        itr = 0
        for ln in f.readlines():

            if "Stop solving model `(default)':" in ln:
                itr += 1
                if itr >= 2:
                    break
            elif 'dual inequality' in ln:
                var = re.findall('Y_[0-9]+', ln)[0]
                val = float(re.findall('[0-9]\.[0-9]+', ln)[0])

                dual_vars[var]['dual_var_value'] = float(val) 
                
                continue
            elif 'Linear relaxation solution' in ln:
                break

    
    df_dual_vars = pd.DataFrame().from_dict(dual_vars, orient='index')
    ser_1 = df_dual_vars[df_dual_vars['serie'] == 1]

    print(ser_1)


    ser_1.to_csv('SERIE1_TEST.csv')
    #ser_1.plot(x='pos_serie', y='dual_var_value')

    #plt.show()

    break