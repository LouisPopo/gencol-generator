import re
import pandas as pd

def compute(file):

    instance_name = '3b_2p5_187'
    nb_ineq = '690'
    input_file_name = 'Networks/Network{}/inputProblem{}_{}_P_TEST_3_inequalities.in'.format(instance_name, instance_name, nb_ineq)
    dual_vars_file_name = 'Networks/Network{}/dualVarsFirstLinearRelaxProblem{}_default.out'.format(instance_name, instance_name)

    # 1. READ LE INPUT PROBLEM FILE
    with open(input_file_name) as f:
        ineqs = []
        for ln in f:
            if ln.startswith('Y_'):
                covers = re.findall(r"\b\w*Cover\w*\b", ln)
                ineqs.append(covers)           
    
    #ineqs = [i for p in ineqs for i in p]
    #print(ineqs)
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

    pi_vals = pd.read_csv(dual_vars_file_name, sep=' ', names=['name', 'pi_value'])
    pi_vals = pi_vals[pi_vals['name'].str.startswith('Cover')]
    pi_vals = dict(zip(pi_vals['name'], pi_vals['pi_value']))
    
    # pour chaque serie, pour chaque paire possible, on regarde si l'inégalitée est bonne
    # i >= i + 1 >= i + 2, etc..
    for serie in series:
        nb_pairs = 0
        nb_pair_error = 0
        for i in range(len(serie) - 1):
            i_val = pi_vals[serie[i]]
            for j in range(i+1, len(serie)):
                j_val = pi_vals[serie[j]]

                if i_val < j_val:
                    nb_pair_error += 1
                
                nb_pairs += 1
        print('{}/{} wrong ({})'.format(nb_pair_error, nb_pairs, nb_pair_error/nb_pairs))

    print('======')

    nb_wrong_ineq = 0
    for serie in series:
        
        for i in range(len(serie) - 1):
            i_val = pi_vals[serie[i]]
            j_val = pi_vals[serie[i+1]]

            if i_val < j_val:
                nb_wrong_ineq += 1

    print('{}/{} wrong ineqs ({})'.format(nb_wrong_ineq, sum([len(s) - 1 for s in series]), nb_wrong_ineq/sum([len(s) for s in series])))


        




if __name__ == '__main__':
    compute(1)