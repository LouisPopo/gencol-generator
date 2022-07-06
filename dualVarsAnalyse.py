import matplotlib.pyplot as plt
import glob
import numpy as np

for instance in glob.glob('RUN1_DUALSVARS/*'):
    
    vals = []

    with open(instance, 'r') as f:

        lines = f.read().splitlines()

        for l in lines :
            
            if 'Cover' in l:

                for s in l.split():

                    if 'e-' in s:
                        vals.append(0)
                    elif '-' in s:
                        vals.append(-int( s.replace('-', '') ) )
                    elif s.isdigit():
                        if int(s) < 850:
                            vals.append(int(s))
                            break

        print(vals)

    w = 2

    plt.hist(vals, edgecolor='black', bins=np.arange(min(vals), max(vals) + w + 0.5, w))
    #plt.hist(vals)
    plt.show()

                