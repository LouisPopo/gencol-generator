# Get all dualVars... files in MdevspGencol and store them in their respective Network folders
# and report files : so we can analyze time


from glob import glob
import os
import shutil

unsolved = []
nb = 0
tot= 0

# retrieving inequalities


for instance_folder in glob('Networks/Network*'):

    nb+=1

    # instance id > 100
    #instance_id = int(instance_folder.split('_')[-1])
    
    #if instance_id < 100:
    #    continue

    ineq_pb_name = None
    for file in glob('{}/*'.format(instance_folder)):
        if '_default.in' in file:
            ineq_pb_name = file.split('/')[2].replace('input', '').replace('.in', '')
    print(ineq_pb_name)

    if ineq_pb_name == None:
        continue

    instance_info = instance_folder.split('/')[1].replace('Network', '')

    #dual_vars_path = '../MdevspGencolTest/dualVarsFirstLinearRelaxProblem{}_default.out'.format(instance_info)
    ineq_dual_vars_path = '../MdevspGencolTest/dualVarsFirstLinearRelax{}.out'.format(ineq_pb_name)
    ineq_report_path = '../MdevspGencolTest/report{}.out'.format(ineq_pb_name)
    ineq_out_path = '../MdevspGencolTest/out{}'.format(ineq_pb_name.replace('Problem', ''))

    if os.path.exists(ineq_dual_vars_path) and os.path.exists(ineq_out_path):

        shutil.copy(ineq_dual_vars_path, instance_folder)
        shutil.copy(ineq_report_path, instance_folder)
        shutil.copy(ineq_out_path, instance_folder)

    else:

        unsolved.append(instance_info)

print('{}/{} instances were unsolved : {}'.format(len(unsolved), nb, unsolved))

          

# for file in glob('../MdevspGencolTest/dualVarsFirstLinearRelax*'):

#     instance_info = file.replace('../MdevspGencolTest/dualVarsFirstLinearRelaxProblem', '').replace('_default.out', '')

#     print(instance_info)

#     network, max_min, seed = instance_info.split('_')

#     print('Veryfing : Networks/Network{}'.format(network))

#     instance_folder = 'Networks/Network{}_{}_{}'.format(network, max_min, seed)

#     if os.path.exists(instance_folder):

#         print('Exists')

#         print('Copying : {} to {}'.format(file, instance_folder))

#         if os.path.exists('../MdevspGencolTest/{}'.format(file)):
#             print('source exists')
        
#         if os.path.exists(instance_folder):
#             print('destination exists')

#         shutil.copy('../MdevspGencolTest/{}'.format(file), instance_folder)


# Certains folders n'auront pas de dual var, car elles n'auront pas été générées

