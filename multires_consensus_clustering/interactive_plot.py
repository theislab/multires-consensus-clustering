import matplotlib
import upsetplot
from plotly.graph_objs import *
import pandas as pd
import numpy as np
import plotly.express as px
import chart_studio.plotly as py
import igraph.layout
from upsetplot import plot
from upsetplot import UpSet
from matplotlib import pyplot as plt

import igraph as ig


def plot_interactive_graph(G):
    """
    Plots a interactive version of the graph, which can display the cells contained in a cluster.

    @param G: The graph to plot, iGraph object graph. Has attributes .vs["name"], .vs["clustering"],
        .vs["cell"], .es["weight"].
    @return:
    """

    name_node = list(G.vs['name'])
    edge_weight = list(G.es["weight"])
    len_name = len(name_node)
    edge_list_tuple = [e.tuple for e in G.es]
    layout_graph = G.layout_auto()

    node_x = [layout_graph[node][0] for node in range(len_name)]
    node_y = [layout_graph[node][1] for node in range(len_name)]
    edge_x = []
    edge_y = []
    for edge in edge_list_tuple:
        edge_x += [layout_graph[edge[0]][0], layout_graph[edge[1]][0], None]
        edge_y += [layout_graph[edge[0]][1], layout_graph[edge[1]][1], None]

    trace_edges = Scatter(x=edge_x, y=edge_y,
                          mode='lines',
                          line=dict(color='rgb(210,210,210)', width=1),
                          text=edge_weight,
                          hoverinfo='text'
                          )
    trace_nodes = Scatter(x=node_x, y=node_y,
                          mode='markers',
                          name='Clusters',
                          marker=dict(symbol='circle',
                                      size=10,
                                      line=dict(color='rgb(50,50,50)', width=0.5),
                                      colorscale='YlGnBu'
                                      ),
                          text=name_node,
                          hoverinfo='text'
                          )

    node_adjacencies = []
    for node, adjacencies in enumerate(G.get_adjacency()):
        node_adjacencies.append(adjacencies.count(1))

    trace_nodes.marker.color = node_adjacencies

    fig = Figure(data=[trace_edges, trace_nodes],
                 layout=Layout(
                     title='multires-consensus-clustering',
                     titlefont_size=20,
                     showlegend=False,
                     hovermode='closest',
                     margin=dict(b=50, l=20, r=20, t=50),
                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                 )
    fig.show()


def bar_chart_nodes(G, data):
    label_list = []

    for merged_vertices in G.vs["cell"]:
        cluster_cell_list = []
        for cell_list in merged_vertices:
            cluster_cell_list = cell_list + cluster_cell_list
        label_list.append(cluster_cell_list)

    cell_names = pd.DataFrame(data["cell"])

    i = 0
    list_df_nodes = []

    for merged_node in label_list:
        values, counts = np.unique(merged_node, return_counts=True)
        counts = counts / len(G.vs[i]["name"])
        node_cell_count = pd.DataFrame({"cell": values, 'merged_node' + ":" + str(i): counts})
        merged_cell_count = pd.concat([cell_names.set_index("cell"), node_cell_count.set_index("cell")], axis=1)
        list_df_nodes.append(merged_cell_count)

        i += 1

    all_cell_counts = pd.DataFrame()
    for cell_df in list_df_nodes:
        all_cell_counts = pd.concat([all_cell_counts,cell_df])

    all_cell_counts = all_cell_counts.groupby(['cell']).sum()
    cell_probability = all_cell_counts.reindex().loc[:, all_cell_counts.reindex().columns != 'cell']

    all_cell_counts[all_cell_counts > 0] = True
    all_cell_counts[all_cell_counts == 0] = False

    column_names = []
    for name_node_cluster in all_cell_counts.columns.values:
        column_names.append(name_node_cluster)
    cell_in_node_size_upset = all_cell_counts.groupby(column_names).size()
    plot(cell_in_node_size_upset)

    plt.show()