# Get all dualVars... files in MdevspGencol and store them in their respective Network folders
# and report files : so we can analyze time


from glob import glob
import os
import shutil

unsolved = []

for instance_folder in glob('Networks/Network*'):

    instance_info = instance_folder.replace('Network', '')

    dual_vars_path = '../MdevspGencolTest/dualVarsFirstLinearRelaxProblem{}_default.out'.format(instance_info)

    if os.path.exists(dual_vars_path):

        shutil.copy(dual_vars_path, instance_folder)

    else:

        unsolved.append(instance_info)

print('Unsolved instances : {}'.format(unsolved))

          

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

