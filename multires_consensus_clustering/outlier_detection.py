import numpy as np
import igraph as ig
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import hdbscan
import multires_consensus_clustering as mcc


def delete_edges_below_threshold(graph, threshold, delete_single_nodes):
    """
    Deletes all edges, of the given graph, below a certain threshold.
    weight
    @param delete_single_nodes: Boolean to delete vertices with degree=0.
    @param graph: The graph, igraph object.
    @param threshold: Float number for the threshold, every edge with weight < threshold is deleted.
    @return: The graph without all edges with edge weight < threshold.
    """

    # delete all edges with less then the threshold
    graph.es.select(weight_le=threshold).delete()

    # delete all vertices with no connection
    if delete_single_nodes:
        graph.vs.select(_degree=0).delete()

    return graph


def delete_small_node_communities(vertex_clustering):
    """
    Deletes communities that are smaller the the average community size.

    @param vertex_clustering: The vertex clustering that creates the different communities on the graph.
    iGraph object: igraph.clustering.VertexClustering.
    @return: Returns the VertexClustering without the small communities.
    """
    subgraph_list = vertex_clustering.subgraphs()
    sum_subgraphs = sum([subgraph.vcount() for subgraph in subgraph_list])
    normalized_subgraph_size = sum_subgraphs / len(subgraph_list)

    vertex_list = []
    for subgraph in subgraph_list:
        if subgraph.vcount() < normalized_subgraph_size:
            vertex_list.append(vertex_clustering.graph.vs.select(name_in=subgraph.vs["name"]))

    vertex_clustering.graph.vs.select(name_in=set(vertex_list)).delete()

    vertex_clustering.graph.simplify(multiple=True, loops=True, combine_edges=max)

    return vertex_clustering.graph


def delete_nodes_with_zero_degree(graph):
    """
    Deletes all nodes without a connection to the rest of the graph.

    @param graph: The graph on which the nodes should be deleted.
    @return: Returns a graph where ever node has degree > 0.
    """

    while graph.vs.select(_degree=0):
        graph.vs.select(_degree=0).delete()

    return graph


def hdbscan_outlier(graph, threshold, plot_on_off):
    """
    Uses the hdbscan density clustering to detect outlier communities in the graph and deletes them.

    @param plot_on_off: Turn the density distribution plot on or off, type Boolean.
    @param threshold: 1-threshold is the quantile above which all connections are deleted.
    @param graph: The graph on which the outliers should be detected. Needs attribute graph.es["weight"].
    @return: The graph without the outlier vertices and all multiple edges combined into single connections by max weight.    """

    # check if density outlier score can be calculated
    if graph.average_path_length(directed=False, unconn=False) != np.inf:
        inverted_weights = [1 - edge_weight for edge_weight in graph.es["weight"]]
        graph.es["weight"] = inverted_weights

        distance_matrix = mcc.create_distance_matrix(graph)
        # distance_matrix = graph.get_adjacency_sparse(attribute="weight")
        clusterer = hdbscan.HDBSCAN(metric="precomputed", min_samples=2, allow_single_cluster=True).fit(distance_matrix)

        # hdbscan outlier detection
        # https://hdbscan.readthedocs.io/en/latest/outlier_detection.html
        threshold = pd.Series(clusterer.outlier_scores_).quantile(1 - threshold)
        outliers = np.where(clusterer.outlier_scores_ > threshold)[0]

        if plot_on_off:
            # hdbscan density plot
            sns.displot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], rug=True)
            plt.show()

            # hdbscan tree plot
            clusterer.condensed_tree_.plot(select_clusters=True,
                                           selection_palette=sns.color_palette('deep', 8))
            plt.show()

            color = ig.drawing.colors.ClusterColoringPalette(2)
            for vertex in graph.vs:
                if vertex.index in outliers:
                    vertex["color"] = color[0]
                else:
                    vertex["color"] = color[1]

        # delete outliers and merge multiple edges and delete loops
        graph.delete_vertices(outliers)
        graph.simplify(multiple=True, loops=True, combine_edges=max)

        # assign weights back to similarity
        inverted_weights = [1 - edge_weight for edge_weight in graph.es["weight"]]
        graph.es["weight"] = inverted_weights

    return graph


def filter_by_node_probability(graph, threshold):
    """
    Calculates the average probability for all cells (probability per cell > 0)
        and deletes all nodes with a probability less than the given threshold.
    @param graph: The graph on which to filter out the nodes, iGraph graph; need graph.vs["probability_df"]
    @param threshold: The threshold below which the vertices are deleted.
    @return: The graph without the vertices.
    """
    vertex_to_delete = []
    # calculates the average probability for every vertex
    for vertex in graph.vs:
        probabilities = vertex["probability_df"]
        probability_vertex = sum(probabilities) / np.count_nonzero(probabilities)

        # if below threshold deletes adds vertices to delete list
        if probability_vertex < threshold:
            vertex_to_delete.append(vertex)

    # if list not empty delete the given vertices
    if vertex_to_delete:
        graph.delete_vertices(vertex_to_delete)

    return graph
