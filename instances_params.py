


network_num = '4b'

max_minutes = [5]

var_modif = 3
nb_bornes = 3
nb_veh = 100                            # Nb. de vehicules disponibles a un depot

nb_instances_per_max_min = 3

fixed_cost = 1000                       # Cout d'un vehicule
nb_veh = 100             

sigma_max = 100000                      # En Wh
speed = 18/60                           # en km/min (en moyenne, sans compter les arrets)
enrgy_km = 1050                         # Energie consommee en Wh par km parcouru
enrgy_w = 11000/60                      # Energie consommee en Wh par minute passee a l'arret

cost_w = 2                              # Cout d'attente par min.
cost_t = 4                              # Cout voyage a vide par min.
delta = 45                              # Temps. max entre la fin d'un trajet et le debut d'un autre

p = 15                                  # Nb. de periodes d'echantillonage pour la recharge

recharge = 15