# va lire les fichiers en sortie et parse jusqua trouver le temps de resolution


with open('duals_inequalities_instances.txt', 'r') as f:
    instances_info_list = f.read().splitlines()



for instance_info in instances_info_list:
    def_info, ineq_info = instance_info.split('/')

    # 'reportProblem{}_default.out'.format(def_info)
    # 'reportProblem{}_{}.out'.format(def_info, ineq_info)

    with open('MdevspGencolTest/reportProblem{}_default.out'.format(def_info)) as f:
        def_report = f.read().splitlines()
        print(def_report[-1])

    with open('MdevspGencolTest/reportProblem{}_{}.out'.format(def_info, ineq_info)) as f:
        ineq_report = f.read().splitlines()
        print(ineq_report[-1])


