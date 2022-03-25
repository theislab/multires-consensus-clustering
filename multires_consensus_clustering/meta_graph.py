from numba import njit
import igraph as ig
import numpy as np
import pandas as pd
import itertools
import multires_consensus_clustering as mcc
import seaborn as sns
from multires_consensus_clustering import binning


def meta_graph(clustering_data, settings_data, bin):
    """
    Uses the Meta Graph script to build the graph from the sc data.

    @param clustering_data: The clustering of the adata file, cells as index and cluster labels for values
        and clustering names as column index
    @param settings_data: The settings used for the clustering_data, number clusters, etc.
    @param bin: The bin from which the clustering should be created, see binning.py
    @return: The meta graph; "graph" an igraph object graph.
    """

    # binning the clusterings in bins close to the given numbers, combines all bins contained in the list
    number_of_clusters_data = sort_by_number_clusters(settings=settings_data, data=clustering_data,
                                                      number_of_clusters_list=bin)

    # builds the graph from the bins
    graph = build_graph(number_of_clusters_data, clustering_data)

    # outlier detection
    if bin < 200:
        graph = mcc.hdbscan_outlier(graph, threshold=0.1, plot_on_off=False)

    # detect and merge communities in the meta graph
    graph = mcc.igraph_community_detection(graph, detection_algorithm="leiden")

    # contract graph clustering into single node
    graph = mcc.contract_graph(graph)

    return graph


def create_and_plot_single_resolution_graph(clustering_data, settings_data, adata_s2d1, bin_number):
    """
    Creates a single resolution meta graph and plots the graph as an interactive plot.
    @param bin_number: The bin from which the meta graph should be created.
    @return: Returns the created meta-graph.
    """
    # single resolution meta graph
    graph = mcc.meta_graph(clustering_data, settings_data, [bin_number])

    # create an interactive plot for the single resolution meta graph
    mcc.interactive_plot(adata_s2d1, clustering_data, graph, create_upsetplot=False,
                         create_edge_weight_barchart=False, layout_option="auto")

    return graph


# relabeling clusters by Isaac
def relabel_clusterings(clusterings: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure each cluster in a set of clusterings has a unique integer ID.

    Each clustering should have a contiguous range of integers for it's cluster's IDs.
    """
    clusterings = clusterings.rank(axis=0, method="dense").astype(int) - 1
    cvals = clusterings.values
    cvals[:, 1:] += (cvals[:, :-1].max(axis=0) + 1).cumsum()
    return clusterings


# jaccrad index computation by Isaac
@njit
def clustering_edges_array(clustering1: "np.ndarray[int]",
                           clustering2: "np.ndarray[int]") -> "list[tuple[int, int, float]]":
    """
    Find jaccard similarities between clusters in two clusterings.

    Params
    ------
    c1, c2
        Clusterings encoded as array of integers. Assumes each cluster has a unique integer id (will be node label in graph).

    Returns
    -------

    Edges in the cluster-cluster graph encoded as tuples of (node1, node2, jaccard similarity).
    """
    edges = []
    offset1 = clustering1.min()
    offset2 = clustering2.min()
    # Because of how I've done unique node names, potentially this
    # could be done in a more generic way by creating a mapping here.
    offset_clusts1 = clustering1 - offset1
    offset_clusts2 = clustering2 - offset2
    # Allocate coincidence matrix
    nclusts1 = offset_clusts1.max() + 1
    nclusts2 = offset_clusts2.max() + 1
    coincidence = np.zeros((nclusts1, nclusts2))
    # Allocate cluster size arrays
    ncells1 = np.zeros(nclusts1)
    ncells2 = np.zeros(nclusts2)
    # Compute lengths of the intersects
    for cell in range(len(clustering1)):
        c1 = offset_clusts1[cell]
        c2 = offset_clusts2[cell]
        # Coincidence matrix stores the size of the intersection of the clusters
        coincidence[c1, c2] += 1
        ncells1[c1] += 1
        ncells2[c2] += 1
    for cidx1, cidx2 in np.ndindex(coincidence.shape):
        isize = coincidence[cidx1, cidx2]
        if isize < 1:
            continue
        # Jaccard similarity is |intersection| / |union|.
        # Here, |union| is calculated as |clustering1| + |clustering2|
        jaccard_sim = isize / (ncells1[cidx1] + ncells2[cidx2] - isize)
        edge = (cidx1 + offset1, cidx2 + offset2, jaccard_sim)
        edges.append(edge)
    return edges


# read data
def read_data(path):
    """
    Reads the data from file and converts it to a pandas df, returns a sample or the complete data.

    :param path: Path of the data.
    :return: Data as a pandas dataframe
    """

    # read the data using pandas
    data = pd.read_table(path)

    return data


def build_graph(clusters, data):
    """
    Builds a meta-graph based on the given data (list of clusterings).

    Assigns each cluster for each clustering a node in the graph and calculates the edges (weighted) using the jaccard index.
    :param data: Pandas dataframe from the single cell data.
    :return: Returns an igraph object Graph.
    """
    # create graph
    G = ig.Graph()

    # all possible combinations of clusters
    combinations_of_clusters = list(itertools.combinations(clusters.columns, 2))

    # create variables
    vertex_cluster_methode, vertex_cells, edge_labels = [], [], []
    vertex_labels = set()

    for cluster_methode in combinations_of_clusters:

        # name of the clustering methode
        cluster_methode_0 = cluster_methode[0]
        cluster_methode_1 = cluster_methode[1]

        # name of the cluster
        cluster_0 = clusters[cluster_methode_0]
        cluster_1 = clusters[cluster_methode_1]

        # compute the jaccard index using Isaacs code, returns a list with start, end and weight of an edge
        edge_list = clustering_edges_array(cluster_0.values, cluster_1.values)

        for edge in edge_list:
            # name of the edges and edge weight
            edge_start = " ".join([cluster_methode_0, ":", str(edge[0])])
            edge_end = " ".join([cluster_methode_1, ":", str(edge[1])])
            edge_weight = edge[2]

            # check if node for cluster_0 is already in the graph
            if edge_start not in vertex_labels:
                G.add_vertices(edge_start)
                vertex_labels.add(edge_start)
                vertex_cluster_methode.append(cluster_methode_0)

                # find all cell names of cluster_methode_0
                cell_data = data["cell"].values[np.where(data[cluster_methode_0] == edge[0])].tolist()
                vertex_cells.append(cell_data)

            # check if node for cluster_1 is already in the graph
            if edge_end not in vertex_labels:
                G.add_vertices(edge_end)
                vertex_labels.add(edge_end)
                vertex_cluster_methode.append(cluster_methode_1)

                # find all cell names of cluster_methode_1
                cell_data = data["cell"].values[np.where(data[cluster_methode_1] == edge[1])].tolist()
                vertex_cells.append(cell_data)

            # check if edge weight is zero
            if edge_weight != 0:
                # add edge to graph
                G.add_edges([(edge_start, edge_end)])
                edge_labels.append(np.round(edge_weight, 5))

    # add edge weight and name to graph
    G.es["weight"] = edge_labels
    G.vs["name"] = list(vertex_labels)
    G.vs["clustering"] = vertex_cluster_methode
    G.vs["cell"] = vertex_cells

    return G


# returns a list of all clusterings closest two the given number of clusters or with the number of cluster (if >= 2)
def sort_by_number_clusters(settings, data, number_of_clusters_list):
    """
    Uses the  binning.bin_n_clusters() function to fit the data into bins of equal size.
    Afterwards turns the number_of_clusters in each bin into a list of lables of the clusters.

    :param settings: Setting data from the sc clustering as a pandas df.
    :param data: Clustering data from the sc data as a pandas df.
    :param number_of_clusters_list: The number of clusters which should be binned together, either an int or a list.
    :return: Returns a list of the clusterings binned by the number of clusters, e.g. [C001, C002, ...].
    """

    list_of_clusterings = []

    # binning function by Luke Zappia, returns list of lists, e.g. [[2], [3,4], ..]
    bins_clusterings = binning.bin_n_clusters(settings["n_clusters"])

    # if number_of_clusters_list == [4,5, .. ] a list of bins return the labels based on the closest bins
    if type(number_of_clusters_list) == list:

        # remove all duplicates
        number_of_clusters_list = list(set(number_of_clusters_list))

        for number_of_clusters in number_of_clusters_list:
            # finds the closest bin to any given number_of_clusters and returns the bins index
            best_bin = min(range(len(bins_clusterings)), key=lambda i:
            abs(bins_clusterings[i][0] + bins_clusterings[i][-1] - 2 * number_of_clusters))

            # selects the bin
            number_of_clusters_closest = bins_clusterings[best_bin]

            # find all clusterings contain in the bin by number_of_clusters
            list_of_clusterings.extend(
                settings.loc[settings['n_clusters'].isin(number_of_clusters_closest)]["id"].values)
    # number_of_clusters_list is a single cluster given as an int or np.int64
    else:
        # convert to int
        number_of_clusters = int(number_of_clusters_list)

        # finds the closest bin to any given number_of_clusters and returns the bins index
        best_bin = min(range(len(bins_clusterings)), key=lambda i:
        abs(bins_clusterings[i][0] + bins_clusterings[i][-1] - 2 * number_of_clusters))

        # selects the bin
        number_of_clusters_closest = bins_clusterings[best_bin]

        # find all clusterings contain in the bin by number_of_clusters
        list_of_clusterings.extend(settings.loc[settings['n_clusters'].isin(number_of_clusters_closest)]["id"].values)

    # return the list with all clusterings
    return data[list_of_clusterings]
