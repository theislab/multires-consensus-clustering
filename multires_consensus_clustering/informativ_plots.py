import numpy as np
import scanpy as sc
import pandas as pd
import multires_consensus_clustering as mcc
import matplotlib.pyplot as plt
from upsetplot import plot


def plot_edge_weights(graph, plot_on_off):
    """
    Create a bar-chart of the edge weight based on the given graph.

    @param plot_on_off: Turn the plot on or off, type: Boolean
    @param graph: The graph, an igraph object, type: graph.
    @return mean_edge_value: The avervage weight of the graph edges. If there are no edges return 0.
    """
    number_edges = graph.ecount()

    # if there are no edges, return 0
    if number_edges == 0:
        return 0

    # else plot barchart of edge weights and return the averge edge weight
    else:
        edge_weights = graph.es["weight"]
        mean_edge_value = sum(edge_weights) / len(edge_weights)

        if plot_on_off:
            # distribution edge weights, histogram with average line
            plt.title("Distribution of edge weights")
            plt.hist(edge_weights, edgecolor='k', bins=40)
            plt.axvline(mean_edge_value, color='k', linestyle='dashed', linewidth=1)
            plt.tight_layout()
            plt.show()

        return mean_edge_value


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

    plt.tight_layout()
    plt.show()


def cell_occurrence_plot(graph, adata, clustering_data, plot):
    """
    Creates a plot of the times each cell occurs in the graph. This plot is shown on the umap plot with the color
        beeing the number of times a cell appers in the merged nodes, summed up.
    @param clustering_data: The clustering data from which the graph is created.
    @param adata: The adata file on which the clustering is based.
    @param graph: The graph from which the cell counts should be generated. iGraph graph, need attribute .vs["cell"]
    """

    cells_list = []

    for cells_vertex in graph.vs["cell"]:
        cells_list.append(sum(cells_vertex, []))

    all_cells_graph = sum(cells_list, [])

    # create df for the probability values
    cell_counts_df = pd.DataFrame()
    cell_counts_df["cell"] = clustering_data["cell"]
    cell_counts_df.set_index("cell", inplace=True)

    # count cell occurrences
    cell_names, cell_counts = np.unique(all_cells_graph, return_counts=True)

    # creates new column in df
    df_probabilities_vertex = pd.DataFrame({'cell': cell_names,
                                            'cell_counts': cell_counts})

    # reset index
    df_probabilities_vertex.set_index("cell", inplace=True)

    # assigns the values to the cells
    cell_counts_df = pd.concat([cell_counts_df, df_probabilities_vertex], axis=1)

    # change NaN values to 0
    cell_counts_df = cell_counts_df.fillna(0)

    # plot cell counts with scanpy and umap
    adata.obs["cell_counts"] = cell_counts_df["cell_counts"]
    if plot:
        plt.tight_layout()
        sc.pl.umap(adata, color=["cell_counts"], show=True)

    return cell_counts_df["cell_counts"].values


def vertex_probabilities_plot(graph):
    """
    Plots a histogram of the probabilities for the nodes in the given graph. The probability of each node is the sum of
        the cell probabilities divided by how many non zero entries there are.
    @param graph: The graph from which the probabilities should be calculated. iGraph graph, need attribute .vs["probability_df"]
    """

    probability_vertex_list = []

    # sum up all edge weights and calculate the average per vertex
    for vertex in graph.vs:
        cell_probabilities = vertex["probability_df"]
        probability_vertex_list.append(sum(cell_probabilities) / np.count_nonzero(cell_probabilities))

    # calculate the overall probability average
    probability_average = sum(probability_vertex_list) / len(probability_vertex_list)

    # plot probability distribution of the vertices as a histogram with an average line
    plt.title("Probability distribution of the nodes")
    plt.hist(probability_vertex_list, range=[0, 1], edgecolor='k', bins=40)
    plt.axvline(probability_average, color='k', linestyle='dashed', linewidth=1)
    plt.tight_layout()
    plt.show()


def probability_umap_plot(df_clusters, adata):
    """
    Display's the probabilities of the cells in the final cluster labels assignment.

    @param df_clusters: The pandas DataFrame created from the multi-res graph [mcc.graph_to_cell_labels_df(graph)].
        This can either plot the original probabilities from the graph or overwrite these probabilities with the new
        uncertainty measure frome the uncertainty_measure_cells(graph, adata) function, which creates a better estimation.
        E.g. df_clusters["probability"].
    @param adata: The adata file on which the mcc function is based.
    """

    # get probably values of the graph, created with the uncertainty_measure_cells function
    estimator = df_clusters["probability"]

    # create an adata label with the new probabilities on the UMAP-plot.
    adata.obs["mcc_cluster_probabilities"] = estimator
    plt.tight_layout()

    # plot UMAP-plot of the probabilities suing the scanpy package
    sc.pl.umap(adata, color=["mcc_cluster_probabilities"], show=True)


def node_level_plot(graph, adata):
    """
    Display's the level of the nodes from which the cells are assigned, in the final cluster label assignment.

    @param graph: The graph from which the final cluster labels are generated.
    @param adata: The adata file on which the mcc function is based.
    """

    # get df of the cluster labels, probabilities and levels from the graph
    df_clusters = mcc.graph_to_cell_labels_df(graph)

    # plot the levels with umap
    adata.obs["mcc_cluster_level"] = df_clusters["level_cluster_label"]
    plt.tight_layout()
    sc.pl.umap(adata, color=["mcc_cluster_level"], show=True)
