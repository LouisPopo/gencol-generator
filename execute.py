import subprocess

import time
nb_cpu = 2
#cpu_free = 3 # 2+1
out_file = open("out", "w")
out_file.close()


list_pb = []

# for i in range(3):
#     for s in ['_S', '']:
#         for n in [1, 5, 10, 25, 50, 100, 200]:
#             list_pb.append("5a_0_460_{}{}_{}".format(n, s, i))
for i in range(3):

    #list_pb.append("4b_0_386_0")

    for nb_in in [100,200,300]:
        for grp_size in [50,100,200,300]:
            if grp_size > nb_in:
                continue
            list_pb.append("4b_0_604_{}_GRP_{}_{}_{}".format(nb_in, int(nb_in/grp_size), grp_size, i))

list_process = []
idx_active = []
for i, pb in enumerate(list_pb) :
    #cpu_free -= 1
    idx_active.append(i)
    with open('out'+pb, 'w') as f:
        list_process.append(subprocess.Popen(['ExecMdevspGencol','Problem'+pb], stdout=f, stderr=f))
    while len(idx_active)>=nb_cpu:
        time.sleep(0.1)
        for j in idx_active:
            done = list_process[j].poll()
            if done is not None:
                idx_active.remove(j)
                #cpu_free += 1
                with open('out', 'a') as out:
                    out.write(f'Done Problem{list_pb[j]} ({len(list_process)-len(idx_active)}/{len(list_pb)})\n')

while len(idx_active)>0:
    time.sleep(0.1)
    for j in idx_active:
        done = list_process[j].poll()
        if done is not None:
            idx_active.remove(j)
            with open('out', 'a') as out:
                out.write(f'Done Problem{list_pb[j]} ({len(list_process)-len(idx_active)}/{len(list_pb)})\n') 