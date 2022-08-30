# given an instance, go read the folder, read dual vars, read input problem, create a graph

from math import floor
from tracemalloc import start
from typing import NewType
import networkx as nx

import plotly.graph_objects as go

import matplotlib.pyplot as plt

instance = '4b_20_0'

# Creating graph

G = nx.DiGraph()

found_arcs = False


positions = []
starts = []
ends = []

x_counts = {}

node_text = []

with open('Networks/Network{}/voyages.txt'.format(instance), 'r') as f:

    for i, line in enumerate(f):

        infos = line.split(';')

        print(infos)

        starts.append(infos[2])
        ends.append(infos[4])

        x_pos = int(int(infos[2]) / 15) # converti en range de 15 mins

        if x_pos not in x_counts:
            x_counts[x_pos] = 0
        
        x_counts[x_pos] = x_counts[x_pos] + 1

        y_pos = x_counts[x_pos]

        positions.append((x_pos, y_pos))

with open('Networks/Network{}/inputProblem{}_default.in'.format(instance, instance), 'r') as f:

    for line in f:

        if line.count('n_') == 1 and line.count('t_') == 1:

            # Node

            node_name = line.split(' ', 1)[0].strip()

            node_text.append(node_name)

            node_nb = int(node_name.replace('n_T', ''))

            node_text[node_nb] += ' : ({},{})'.format(starts[node_nb], ends[node_nb])

            # print('Node name is : {}.'.format(node_name))

            G.add_node(node_name, pos=positions[node_nb])

        if line.count('n_') >= 2:

            # Ligne d'un arc entre deux trips

            #print(line)
            source, destination, cost, rest = line.split('[', 1)[0].split(' ')

            # print('Source : {}, Destination : {}, cost : {}'.format(source, destination, cost))

            G.add_weighted_edges_from([(source, destination, cost)])


dual_variables_values = []

with open('Networks/Network{}/dualVarsFirstLinearRelaxProblem{}_default.out'.format(instance, instance), 'r') as f:

    for line in f:
        if 'Cover' in line:

            dual_var_value = line.split(' ')[1].strip()

            #print('{} : {}'.format(dual_var_value, round(float(dual_var_value))))

            trip_nb = int(line.split(' ')[0].replace('Cover_T', ''))

            int_val = round(float(dual_var_value))

            if int_val >= 900:
                int_val = 100

            node_text[trip_nb] += '\n{}'.format(int_val)

            dual_variables_values.append(int_val)

edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines',
    )

node_x = []
node_y = []
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='RdBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

# node_adjacencies = []
# node_text = []
# for node, adjacencies in enumerate(G.adjacency()):
#     node_adjacencies.append(len(adjacencies[1]))
#     node_text.append('# of connections: '+str(len(adjacencies[1])))

node_trace.marker.color = dual_variables_values
node_trace.text = node_text

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.show()