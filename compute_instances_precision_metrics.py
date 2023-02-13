import re
import pandas as pd
from glob import glob

pd.options.mode.chained_assignment = None  # default='warn'
# this file computes instances metrics:
# model accuracy in general (ALL predictions made)
# the accuracy if we have a treshold of 0.2, 0.25, 0.3, 0.35

def compute_model_accuracies(min_id):
    # take a network (instance)
    # get predictions
    # compute overall accuracy
    # compute accuracy over different treshold

    results = []

    for instance_folder in glob('Networks/Network*'):
        instance = instance_folder.replace('Networks/Network','')

        instance_id = int(instance.split('_')[-1])

        if instance_id < min_id:
            continue
    

        df_predictions = pd.read_csv('{}/inequalities_predictions.csv'.format(instance_folder))
        df_predictions = df_predictions[df_predictions['A'] != df_predictions['B']]

        ZEROES = len(df_predictions[df_predictions['real'] == 0.0])
        ONES = len(df_predictions[df_predictions['real'] == 1.0])

        for tresh in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:

            # ceux qu'on prédit "vrai"
            df_above = df_predictions[df_predictions['pred'] > (1 - tresh)]
            nb_above_pred = len(df_above)
            nb_above_real = len(df_above[df_above['real'] == 1.0])
            df_above['A_B'] = df_above['A'] + '_' + df_above['B']
            counted_pairs = set(df_above['A_B'].unique())

            # on doit s'assurer de ne pas prendre les mêmes predictions
            df_under = df_predictions[df_predictions['pred'] < tresh]
            df_under['A_B'] = df_under['B'] + '_' + df_under['A'] # ATTENTION, ICI ON SWITCH
            df_under = df_under[~df_under['A_B'].isin(counted_pairs)]
            nb_under_pred = len(df_under)
            nb_under_real = len(df_under[df_under['real'] == 0.0])

            above_acc = nb_above_real / nb_above_pred
            under_acc = nb_under_real / nb_under_pred

            # print('Above nb pred :  {}. Under nb pred : {}'.format(nb_above_pred, nb_under_pred))
            # print('Above nb real :  {}. Under nb real : {}'.format(nb_above_real, nb_under_real))
            # print('Above acc :      {}. Under acc :     {}'.format(above_acc, under_acc))
            
            tot_pred = nb_above_pred + nb_under_pred
            tot_real = nb_above_real + nb_under_real
            tot_acc = tot_real / tot_pred

            #print('Tot acc : {}'.format(tot_acc))

            results.append([instance, tresh, nb_above_pred, nb_above_real, above_acc, nb_under_pred, nb_under_real, under_acc, tot_acc])

        print('Done for : {}'.format(instance))

    df_results = pd.DataFrame(results, columns=[
        'instance', 
        'tresh',
        'nb above tresh pred',
        'nb above tresh real',
        'above acc',
        'nb under tresh pred',
        'nb under tresh real',
        'under acc',
        'tot acc'
        ])

    df_results.to_csv('instances_metrics.csv')

if __name__ == '__main__':

    min_id = 175

    compute_model_accuracies(min_id)