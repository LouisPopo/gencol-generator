# Get all dualVars... files in MdevspGencol and store them in their respective Network folders

from glob import glob
import os
import shutil

for file in glob('../MdevspGencolTest/dualVarsFirstLinearRelax*'):

    

    instance_info = file.replace('../MdevspGencolTest/dualVarsFirstLinearRelaxProblem', '').replace('_default.out', '')

    print(instance_info)

    network, max_min, seed = instance_info.split('_')

    if os.path.exists('Networks/Network{}'.format(network)):

        print('Exists')

        shutil.copy('../MdevspGencolTest/{}'.format(file), 'Networks/Network{}'.format(network))

