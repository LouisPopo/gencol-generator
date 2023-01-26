# va chercher les infos de toutes les instances résolues et produit un csv :

# parametres, nombre de trajets, temps de resolution, val. sol. optimale, nombre de bus utilisés, etc. 

from glob import glob
import os
import shutil
import pandas as pd
import sys

def get_report_file_infos(file_path):
    with open(file_path, 'r') as f:

        txt_lines = f.readlines()
        
        time_line = txt_lines[3]

        time = float(time_line[time_line.find("(")+1:time_line.find(")")].split(' ')[-1])
        
        obj_val_line = txt_lines[6]

        obj_val = float(obj_val_line.split(':')[1].split(' ')[-1])

        col_gen_iterations_line = txt_lines[21]

        col_gen_iterations = int(col_gen_iterations_line.split(':')[1].strip())

        mem_use_line = txt_lines[55]

        tot_mem_use = float(mem_use_line.split(':')[1].strip())
    
    return time, obj_val, col_gen_iterations, tot_mem_use

def get_out_file_infos(file_path):
    with open(file_path, 'r') as f:

        gotV0, gotV1 = False, False
        nb_Veh_D0 = -1
        nb_Veh_D1 = -1

        for line in f:

            if 'Veh_D0' in line:

                nb_Veh_D0 = float(line.split(' ')[4])

                gotV0 = True
            
            elif 'Veh_D1' in line:

                nb_Veh_D1 = float(line.split(' ')[4])

                gotV1 = True

            if gotV0 and gotV1 :
                break
    return nb_Veh_D0, nb_Veh_D1

def create_analysis_csv(suffix, output_name): 
    results = []
    for instance_folder in glob('Networks/Network*'):

        # instance_info = instance_folder.split('/')[1].replace('Network', '')
        ineq_problem_name = None
        for file in glob('{}/*'.format(instance_folder)):
            if suffix in file and 'input' in file:
                ineq_problem_name = file.split('/')[2].replace('input', '').replace('.in', '')
        if ineq_problem_name == None:
            continue
           
        
        instance_info = ineq_problem_name.replace('Problem', '').split('_')
        network = instance_info[0]
        max_min = instance_info[1]
        seed = instance_info[2]
        inequalities = instance_info[3]

        def_problem_name = 'Problem{}_{}_{}_default'.format(network, max_min, seed)

        dual_vars_file_path = '{}/dualVarsFirstLinearRelax{}.out'.format(instance_folder, ineq_problem_name)
        
        ineq_report_file_path = '{}/report{}.out'.format(instance_folder, ineq_problem_name)
        def_report_file_path = '{}/report{}.out'.format(instance_folder, def_problem_name)
        
        ineq_out_file_path = '{}/out{}'.format(instance_folder, ineq_problem_name.replace('Problem', ''))
        def_out_file_path = '{}/out{}'.format(instance_folder, def_problem_name.replace('Problem', ''))

        trips_file_path = '{}/voyages.txt'.format(instance_folder)

        # df_preds = pd.read_csv('{}/inequalities_predictions.csv'.format(instance_folder))

        # good_sure_predictions = len(df_preds[
        #     (
        #         (df_preds['pred'] >= 0.65) &
        #         (df_preds['real'] == 1.0)
        #     ) 
        #     |
        #     (
        #         (df_preds['pred'] <= 0.35) &
        #         (df_preds['real'] == 0.0)
        #     )
        # ])
        
        # nb_sure_predictions = len(df_preds[
        #     (df_preds['pred'] >= 0.65)
        #     |
        #     (df_preds['pred'] <= 0.35)
        # ]
        # )

        # model_acc = good_sure_predictions/nb_sure_predictions

        #def_report_file_path = '{}/reportProblem{}_{}_{}_default.out'.format(instance_folder, network, max_min, seed)

        print(def_report_file_path)
        print(ineq_report_file_path)

        if os.path.exists(ineq_report_file_path) and os.path.exists(ineq_out_file_path) and os.path.exists(def_report_file_path):

            ineq_time, ineq_obj_val, ineq_col_gen_iterations, ineq_tot_mem_use = get_report_file_infos(ineq_report_file_path)
            def_time, def_obj_val, def_col_gen_iterations, def_tot_mem_use = get_report_file_infos(def_report_file_path)
            
            ineq_nb_veh_0, ineq_nb_veh_1 = get_out_file_infos(ineq_out_file_path)
            def_nb_veh_0, def_nb_veh_1 = get_out_file_infos(def_out_file_path)

            with open(trips_file_path, 'r') as f:

                nb_trips = len(f.readlines())

            results.append([
                network, 
                max_min, 
                int(seed), 
                int(inequalities), 
                nb_trips, 
                def_nb_veh_0,
                def_nb_veh_1,
                def_obj_val,
                def_time,
                def_col_gen_iterations,
                def_tot_mem_use,
                ineq_nb_veh_0,
                ineq_nb_veh_1,
                ineq_obj_val,
                ineq_time,
                ineq_col_gen_iterations,
                ineq_tot_mem_use
            ])
        else:
            print('no exists')
    df_results = pd.DataFrame(
        results,
        columns=[
            'net', 
            'max_min', 
            'seed', 
            'nb_inequalities', 
            'nb_trips', 
            'def_nb_veh_D0', 
            'def_nb_veh_D1', 
            'def_obj_val', 
            'def_sol_time', 
            'def_col_gen_iter', 
            'def_tot_mem_use', 
            'ineq_nb_veh_D0', 
            'ineq_nb_veh_D1', 
            'ineq_obj_val', 
            'ineq_sol_time', 
            'ineq_col_gen_iter', 
            'ineq_tot_mem_use'
            ]
        ) 

    df_results.sort_values(['max_min', 'seed'], inplace=True)

    df_results.to_csv('instances_resolutions_metrics_{}.csv'.format(output_name), index=False)       
    
if __name__ == '__main__':

    if len(sys.argv) <= 2:
        print('Missing arguments : suffix or output_name')
        sys.exit()
    
    suffix = sys.argv[1]
    output_name = sys.argv[2]

    create_analysis_csv(suffix, output_name)