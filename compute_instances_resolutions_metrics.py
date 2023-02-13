from glob import glob
import re
import pandas as pd
import sys

# given one file extansion, get those outExtansion files

def get_metrics(suffix, min_id):

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
        for f in glob('{}/*'.format(instance_folder)):

            if 'report' in f and 'default' in f:
                def_report_file = f
            if 'out' in f and suffix in f and '.' not in f:
                inst_out_file = f

        if def_report_file is None or inst_out_file is None:
            print('{} is missing a file'.format(instance_folder))
            continue

        nb_instances += 1

        def_time = 0
        with open(def_report_file, 'r') as f:
            for l in f:
                if 'Entire solving process' in l:
                    def_time = float(re.findall('\d+\.\d+', l)[0])
                if 'Lower bound after cuts' in l:
                    def_obj_val = float(re.findall('\d+\.\d+', l)[0])
        
    
        with open(inst_out_file, 'r') as f:
            lines = f.readlines()
        stop_line = "Stop solving model `(default)':"
        lines_before = []
        for i, line in enumerate(lines):
            if "Stop solving model `(default)':" in line:
                if i > 0:
                    lines_before.append(lines[i-1].strip())

        if len(lines_before) <= 1:
            print('No lines before : {}'.format(inst_out_file))

        for i,l in enumerate(lines_before):
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", l)
            numbers =  [float(x) if '.' in x else int(x) for x in numbers]
            time = numbers[2] + numbers[5]
            obj_val = numbers[6]

            speedup = def_time / time
            gap = (def_obj_val - obj_val)/def_obj_val
            # print(obj_val)
            # print('{} , {}%'.format(speedup, gap))

            results.append([net, max_min, seed, i+1, time, obj_val, speedup, gap])

        #print('Done with : {}'.format(instance))
    
    df_results = pd.DataFrame(results, columns=[
        'net', 
        'max_min', 
        'seed', 
        'iteration',
        'time',
        'obj_val',
        'speedup',
        'gap'
        ])

    df_results.sort_values(['max_min', 'seed', 'iteration'], inplace=True)
    
    df_results.to_csv('instances_resolutions_metrics_{}.csv'.format(suffix), index=False)
    
    print('Got : {} instances'.format(nb_instances))

def get_coverings(suffix, min_id):

    results = []
    # instance , %>1, %=1, %<1

    nb_instances = 0

    for instance_folder in glob('Networks/Network*'):
        
        instance = instance_folder.replace('Networks/Network','')

        instance_id = int(instance.split('_')[-1])

        #print('Doing : {}'.format(instance_id))

        if instance_id < min_id:
            continue

        print(instance_folder)

        net, max_min, seed = instance.split('_')

        inst_out_file = None
        for f in glob('{}/*'.format(instance_folder)):

            if 'out' in f and suffix in f and '.' not in f:
                inst_out_file = f
                print(f)

        if inst_out_file is None:
            #print('{} is missing a file'.format(instance_folder))
            continue

        nb_instances += 1

        
        covering = dict()

        current_val = 0
        with open(inst_out_file, 'r') as f:
            for l in f:
                if 'net_D1' in l and '{' in l:
                    vals = l.strip().split(' ')
                    current_val = float(vals[2])

                if ' t_' in l and 'n_T' in l and 'o_D' not in l and 'k_D' not in l and 'cost' not in l:
                    node = l.strip().split(' ')[0]
                    if node == 'r_Not_Rch;' or node == 'Sink':
                        print(l)
                    if node not in covering:
                        covering[node] = 0
                    covering[node] = covering[node] + current_val

        df_cov = pd.DataFrame(covering.items(), columns=['Task', 'Cover'])

        more_one = len(df_cov[df_cov['Cover'] >= 1.1])/len(df_cov)
        exact_one = len(df_cov[(df_cov['Cover'] < 1.1) & (df_cov['Cover'] > 0.9)])/len(df_cov)
        less_one = len(df_cov[df_cov['Cover'] <= 0.9])/len(df_cov)

        #df_cov.sort_values(by=['Cover'], ascending=False, inplace=True)

        #print(df_cov) 
        results.append([net, max_min, seed, more_one, exact_one, less_one])  

    df_results = pd.DataFrame(results, columns=['net', 'max_min', 'seed', '>1', '=1', '<1'])
    df_results.sort_values(by=['max_min', 'seed'], inplace=True)
    
    df_results.to_csv('coverings_{}.csv'.format(suffix), index=False)  

   

                
        




if __name__ == '__main__':

    if len(sys.argv) <= 2:
        print('Missing arguments : suffix or id')
        sys.exit()
    
    suffix = sys.argv[1]
    min_id = int(sys.argv[2])

    get_coverings(suffix, min_id)
