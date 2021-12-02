from plotly.graph_objs import *
import chart_studio.plotly as py
import igraph.layout

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
