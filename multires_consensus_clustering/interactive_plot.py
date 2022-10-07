import igraph
from upsetplot import plot
from matplotlib import pyplot as plt
import bokeh
from bokeh.plotting import figure
from bokeh.models import GraphRenderer, HoverTool, ColumnDataSource, Circle, ColorBar
from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,
                          MultiLine, NodesAndLinkedEdges, Plot, Range1d, TapTool)
from bokeh.io import show, output_file
from bokeh.transform import linear_cmap, LogColorMapper
from bokeh.palettes import plasma, Plasma256
from pathlib import Path
from matplotlib import cm
from matplotlib import colors as col
import multires_consensus_clustering as mcc
import pandas as pd
import base64
from io import BytesIO
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import scanpy as sc
import igraph as ig
import numpy as np

HERE = Path(__file__).parent.parent


def interactive_plot(adata_s2d1, clustering_data, graph, create_upsetplot, create_edge_weight_barchart, layout_option):
    """
    Uses the adata to build an interactive plot with bokeh.

    @param clustering_data: Th clustering data as a pandas dataframe.
    @param adata_s2d1: The selected adata data set.
    @param layout_option: The layout for the graph, can be "hierarchy" -> layout based on vertex level.
        Else: iGraph auto layout
    @param graph: The graph on which the plot is based.
    @param create_upsetplot: Boolean variable to turn the upsetplot on or off; "True"/"False"
    @param create_edge_weight_barchart: Boolean variable to turn the edge weight barchart on or off; "True"/"False"
    """

    # creates a pandas df with the probabilities of a cell being in a specific node
    # if the graph is a mulit graph, the nodes already have the probability as an attribute
    if "probability_df" in graph.vs.attribute_names():
        df_cell_probability = pd.DataFrame()
        i = 0

        # the first node also has the cell index of the probability as an attribute
        df_cell_probability["cell"] = graph.vs[0]["cell_index"]

        # convert vertex attributes back to df
        for vertex in graph.vs:
            df_cell_probability["merged_node_" + str(i)] = vertex["probability_df"]
            i += 1

        # replace index
        df_cell_probability.set_index("cell", inplace=True)

    # otherwise the probability df is created
    else:
        df_cell_probability = mcc.graph_nodes_cells_to_df(graph, clustering_data)

    # create an bar-chart with all edge weights
    if create_edge_weight_barchart:
        mcc.plot_edge_weights(graph)

    # create an upsetplot for the data
    if create_upsetplot:
        mcc.upsetplot_graph_nodes(df_cell_probability)

    # create the cluster plots from the adata and adds the images to the graph
    graph = mcc.umap_plot(df_cell_probability, adata_s2d1, graph)

    # plots an interactive graph using bokeh and an upset plot showing how the cells are distributed
    mcc.plot_interactive_graph(graph, df_cell_probability, layout_option)


def plot_interactive_graph(G, df_cell_probability, layout_option):
    """
    Plots a interactive version of the graph, which can display the cells contained in a cluster.

    @param layout_option: Either "hierarchy" for a tree like plot, iGraph vertices need attribute G.vs["level"].
        Otherwise uses the auto layout of iGraph.
    @param df_cell_probability: Pandas Dataframe with a probability value for each cell for being in a merged node.
    @param G: The render_graph to plot, iGraph object graph. Has attributes .vs["name"], .vs["clustering"],
        .vs["cell"], .es["weight"].
    @return:
    """

    # basic properties of the graph G
    number_of_nodes = G.vcount()
    edge_list_tuple = [e.tuple for e in G.es]

    # iGraph layout using layout_option
    if layout_option == "hierarchy":
        if "level" in G.vs.attributes():
            layout_graph = G.layout_sugiyama(layers=G.vs["level"])
        else:
            # catch no attribute
            print("Graph has no attribute: level")
            layout_graph = G.layout_auto()

    # auto layout for any other input
    else:
        layout_graph = G.layout_auto()

    # render_graph attributes for each node/edge
    name_node = list(G.vs['name'])
    clustering_node = list(G.vs["clustering"])
    plot_node = G.vs["img"]
    edge_weight = list(G.es["weight"])

    # interactive plot attributes
    number_clusters_combined = [len(clusters_combined) for clusters_combined in name_node]
    number_clustering_combined = [len(clusterings_combined) for clusterings_combined in clustering_node]
    number_cells_node = df_cell_probability.astype(bool).sum(axis=0).values

    # (x,y)-coordinates for each node
    node_x = [layout_graph[node][0] for node in range(number_of_nodes)]
    node_y = [layout_graph[node][1] for node in range(number_of_nodes)]

    # creates an edge list for all edges in G
    edge_x = []
    edge_y = []
    for edge in edge_list_tuple:
        edge_x.append(edge[0])
        edge_y.append(edge[1])

    # list of nodes in the graph G
    node_indices = list(range(number_of_nodes))

    # initialize the plot and the render_graph
    plot_graph = figure(title="Meta Graph")

    render_graph = GraphRenderer()

    # add attributes to bokeh network render_graph
    # node data
    render_graph.node_renderer.data_source.data["number_cells_node"] = number_cells_node
    render_graph.node_renderer.data_source.data["number_clusters_combined"] = number_clusters_combined
    render_graph.node_renderer.data_source.data["number_clustering_combined"] = number_clustering_combined
    render_graph.node_renderer.data_source.data["index"] = node_indices
    render_graph.node_renderer.data_source.data['name'] = name_node
    render_graph.node_renderer.data_source.data['clustering'] = clustering_node
    render_graph.node_renderer.data_source.data['img'] = plot_node
    # edge data
    if edge_x:
        render_graph.edge_renderer.data_source.data["weight"] = edge_weight

    # add vertex information with hoverTool
    node_hover = HoverTool(
        tooltips=[
            ("img", "@img{safe}"),
            ("component_id:", "@index"),
            ("# clusters:", "@number_clusters_combined"),
            ("# clusterings:", "@number_clustering_combined"),
            ("# cells:", "@number_cells_node")
        ],
        attachment="vertical",
    )
    plot_graph.add_tools(node_hover)
    # assign color and size to nodes in network render_graph
    render_graph.node_renderer.glyph = Circle(size=20,
                                              fill_color=linear_cmap('number_cells_node',
                                                                     plasma(256),
                                                                     min(number_cells_node), max(number_cells_node)))

    # add the edges to the network graph, if there are any
    if edge_x:
        render_graph.edge_renderer.data_source.data = dict(
            start=edge_x,
            end=edge_y)

        # edge line width based on the edge weight
        dict_edge_weight = dict(zip(zip(edge_x, edge_y), edge_weight))
        line_width = [dict_edge_weight[edge] * 10 for edge in zip(render_graph.edge_renderer.data_source.data["start"],
                                                                  render_graph.edge_renderer.data_source.data["end"])]
        render_graph.edge_renderer.data_source.data["line_width"] = line_width
        render_graph.edge_renderer.glyph.line_width = {'field': 'line_width'}

    # create graph layout according to bokeh Visualizing network graphs documentation
    graph_layout = dict(zip(node_indices, zip(node_x, node_y)))

    # use the layout provided from bokeh to create a layout,
    # as used in the bokeh Visualizing network graphs documentation
    render_graph.layout_provider = bokeh.models.graphs.StaticLayoutProvider(graph_layout=graph_layout)

    # render the render_graph
    plot_graph.renderers.append(render_graph)

    # check if there are more then one node
    if min(number_cells_node) != max(number_cells_node):
        color_mapper = LogColorMapper(palette=Plasma256, low=min(number_cells_node), high=max(number_cells_node))
        color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12)
        plot_graph.add_layout(color_bar, 'right')
    else:
        max_legende_value = max(len(df_cell_probability), 10000)
        color_mapper = LogColorMapper(palette=Plasma256, low=1, high=max_legende_value)
        color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12)
        plot_graph.add_layout(color_bar, 'right')

    # create file for plot, type: .html
    output_file(HERE / "plots" / "MetaGraph.html")

    # show plot
    show(plot_graph)


def umap_plot(df_cell_probability, adata, graph):
    """
    Uses the umap from scanpy to plot the probability of the cells being in one node. These range from 0 to 1.
    Saves the plot in directory plots und the node_names.

    @param df_cell_probability: Pandas dataframe with all cells from the clustering (as rows) and merged nodes as columns,
        values of the dataframe are the probabilities of the cell being in the merged node, range 0 to 1.
    @param adata: The original adata set, used for the layout of the plot.
    """

    # creates a plot for every merged node in the dataframe and adds the plot to the graph under G.vs["img"]
    # uses the code from Isaac for the encoding of the image
    # https://github.com/ivirshup/constclust/blob/6b7ef2773a3332beccd1de8774b16f3727321510/constclust/clustree.py#L223
    plot_list = []
    for columns in df_cell_probability.columns:
        adata.obs['probability_cell_in_node'] = df_cell_probability[columns]
        file = columns + ".png"
        plot = sc.pl.umap(adata, color='probability_cell_in_node', show=False)
        with BytesIO() as buf:
            plot.figure.savefig(buf, format="png", dpi=50)
            buf.seek(0)
            byte_image = base64.b64encode(buf.read())
        encode_image = byte_image.decode("utf-8")
        plot_list.append(f'<img src="data:image/png;base64,{encode_image}"/>')
        plt.close()
    graph.vs["img"] = plot_list

    return graph
