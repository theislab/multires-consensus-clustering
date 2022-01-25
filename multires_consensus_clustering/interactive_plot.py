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

HERE = Path(__file__).parent.parent


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

    # assign color and form to nodes in network render_graph
    render_graph.node_renderer.glyph = Circle(size=20,
                                              fill_color=linear_cmap('number_cells_node',
                                                                     plasma(len(number_cells_node)),
                                                                     min(number_cells_node), max(number_cells_node)))

    # add the edges to the network graph
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

    # use the layoutprovider from bokeh to create a layout,
    # as used in the bokeh Visualizing network graphs documentation
    render_graph.layout_provider = bokeh.models.graphs.StaticLayoutProvider(graph_layout=graph_layout)

    # render the render_graph
    plot_graph.renderers.append(render_graph)

    color_mapper = LogColorMapper(palette="Plasma256", low=min(number_cells_node), high=max(number_cells_node))
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12)
    plot_graph.add_layout(color_bar, 'right')

    # create file for plot, type: .html
    output_file(HERE / "plots" / "MetaGraph.html")

    # show plot
    show(plot_graph)


def upsetplot_graph_nodes(df_cell_probability):
    """
    Creates an upsetplot showing the cells in each merged node and the cells shared by the nodes.

    @param df_cell_probability: Pandas dataframe with cells and probabilities for each merged node.
    """

    df_cell_probability = df_cell_probability.groupby(['cell']).sum()

    df_cell_probability[df_cell_probability > 0] = True
    df_cell_probability[df_cell_probability == 0] = False

    column_names = []
    for name_node_cluster in df_cell_probability.columns.values:
        column_names.append(name_node_cluster)
    cell_in_node_size_upset = df_cell_probability.groupby(column_names).size()
    plot(cell_in_node_size_upset)

    plt.show()
