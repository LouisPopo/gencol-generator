# Get all dualVars... files in MdevspGencol and store them in their respective Network folders

from glob import glob
import os
import shutil

for file in glob('../MdevspGencolTest/dualVarsFirstLinearRelax*'):

    instance_info = file.replace('../MdevspGencolTest/dualVarsFirstLinearRelaxProblem', '').replace('_default.out', '')

    print(instance_info)

    network, max_min, seed = instance_info.split('_')

    print('Veryfing : Networks/Network{}'.format(network))

    instance_folder = 'Networks/Network_{}_{}_{}'.format(network, max_min, seed)

    if os.path.exists(instance_folder):

        print('Exists')

        shutil.copy('../MdevspGencolTest/{}'.format(file), instance_folder)

