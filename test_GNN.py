import dgl
import torch as th
import matplotlib.pyplot as plt
import networkx as nx

graph_data = {
   ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
   ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])),
   ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
}

g = dgl.heterograph(graph_data)

print(g.nodes('disease'))