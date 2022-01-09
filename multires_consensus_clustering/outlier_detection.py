import numpy as np
import igraph as ig
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import hdbscan
import multires_consensus_clustering as mcc


def min_cuts(graph):
    """
    Function for the min-cut algorithm. Separates the graph in two community by separating the edges with
     lowest weight. Repeats the process until condition for max separations is met.


    @param graph: The Graph on which the min-cut algorithm is performed.
    @return: Returns the cut graph.
    """
    weight_normalized = np.sum(graph.es["weight"]) / graph.ecount()
    size_partition_1 = 0
    size_partition_2 = 1
    cut_weight = weight_normalized - 1
    while weight_normalized > cut_weight:
        cut = graph.mincut(source=None, target=None, capacity="weight")
        graph.delete_vertices(cut.partition[0])
        size_partition_1 = len(cut.partition[0])
        size_partition_2 = len(cut.partition[1])
        cut_weight = cut.value
        print(cut_weight)
    return graph


def delete_edges_below_threshold(graph, threshold):
    """
    Deletes all edges, of the given graph, below a certain threshold.

    @param graph: The graph, igraph object.
    @param threshold: Float number for the threshold, every edge with weight < threshold is deleted.
    @return: The graph without all edges with edge weight < threshold.
    """

    for edge in graph.es:
        if edge["weight"] < threshold:
            graph.delete_edges([edge.source, edge.target])

    # delete all vertices with no connection
    graph.vs.select(_degree=0).delete()

    return graph


def delete_small_node_communities(vertex_clustering):
    """
    Deletes communities that are smaller the the average community size.

    @param vertex_clustering: The vertex clustering that creates the diffrent communties on the graph.
    iGraph object: igraph.clustering.VertexClustering.
    @return: Returns the VertexClustering without the small communities.
    """
    subgraph_list = vertex_clustering.subgraphs()
    sum_subgraphs = sum([subgraph.vcount() for subgraph in subgraph_list])
    normalized_subgraph_size = sum_subgraphs / len(subgraph_list)
    ig.plot(vertex_clustering)
    # print(vertex_clustering.graph.vcount())
    vertex_list = []
    for subgraph in subgraph_list:
        if subgraph.vcount() < normalized_subgraph_size:
            vertex_list.append(vertex_clustering.graph.vs.select(name_in=subgraph.vs["name"]))

    print(set(vertex_list))
    vertex_clustering.graph.vs.select(name_in=set(vertex_list)).delete()

    vertex_clustering.graph.simplify(multiple=True, loops=True, combine_edges=max)
    # print(vertex_clustering.graph.vcount())
    ig.plot(vertex_clustering)
    return vertex_clustering


def hdbscan_outlier(graph, threshold, plot_on_off):
    """
    Uses the hdbscan density clustering to detect outlier communities in the graph and deletes them.

    @param plot_on_off: Turn the density distribution plot on or off, type Boolean.
    @param threshold: 1-threshold is the density above which all connections are deleted.
    @param graph: The graph on which the outliers should be detected. Needs attribute graph.es["weight"].
    @return: The graph without the outlier vertices and all multiple edges combined into single connections by max weight.    """
    inverted_weights = [1 - edge_weight for edge_weight in graph.es["weight"]]
    graph.es["weight"] = inverted_weights

    distance_matrix = mcc.create_distance_matrix(graph)
    clusterer = hdbscan.HDBSCAN(metric="precomputed").fit(distance_matrix)

    if plot_on_off:
        # hdbscan density plot
        sns.displot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], rug=True)
        plt.show()

    # hdbscan outlier detection
    # https://hdbscan.readthedocs.io/en/latest/outlier_detection.html
    threshold = pd.Series(clusterer.outlier_scores_).quantile(1 - threshold)
    outliers = np.where(clusterer.outlier_scores_ > threshold)[0]

    graph.delete_vertices(outliers)
    graph.simplify(multiple=True, loops=True, combine_edges=max)

    inverted_weights = [1 - edge_weight for edge_weight in graph.es["weight"]]
    graph.es["weight"] = inverted_weights

    return graph
