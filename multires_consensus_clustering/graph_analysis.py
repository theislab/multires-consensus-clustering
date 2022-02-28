import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from multires_consensus_clustering import meta_graph as mg
import itertools
import multires_consensus_clustering as mcc
import seaborn as sns
import sklearn
import hdbscan
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import itertools
import networkx as nx
import seaborn as sns


def igraph_community_detection(G, detection_algorithm):
    """
    Function for community detection. Uses the igraph community detection functions to partition the graph
     in community based on edge weight. Merges the community to one node, by combing all attributes in a list and
     by mean edge weight.

    @param detection_algorithm: Name of the algorithm on which the community detection is based.
        Can be "fast_greedy", "newman2006", louvian" or "all" to get a list algorithm clusterings.
    @param G: The graph on which to detect communities.
    @return: Returns the merged graph based on community.
    """

    # no match case in python <3.10, therefore if/elif
    if detection_algorithm == "fast_greedy":
        # fast_greedy graph community detection
        graph = ig.Graph.community_fastgreedy(G, weights="weight").as_clustering()
        # ig.plot(graph)
        return graph

    elif detection_algorithm == "newman2006":
        # newman2006 graph community detection, more community then the others
        graph = ig.Graph.community_leading_eigenvector(G, weights="weight")
        # ig.plot(graph)
        return graph

    elif detection_algorithm == "louvain":
        # louvain methode for graph community detection
        graph = ig.Graph.community_multilevel(G, weights="weight")
        # ig.plot(graph)
        return graph

    elif detection_algorithm == "leiden":
        # leiden methode for graph community detection, improvement of the louvain methode
        # https://arxiv.org/abs/1810.08473
        graph = ig.Graph.community_leiden(G, weights="weight", objective_function="modularity", n_iterations=-1)
        # ig.plot(graph)
        return graph


def contract_graph(graph):
    """
    Contracts the clustered vertices of a graph into a single node using igraph function cluster_graph().
    Merges attributes into list and combines edges by max edge weight.

    @param graph: A vertex_clustering of a igraph network graph.
    @return: The contracted graph, igraph graph object.
    """

    # contract graph
    graph = graph.cluster_graph(combine_vertices=list, combine_edges=max)

    return graph


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
            plt.hist(edge_weights, edgecolor='k', bins=40)
            plt.axvline(mean_edge_value, color='k', linestyle='dashed', linewidth=1)
            plt.show()

        return mean_edge_value


def hdbscan_community_detection(graph):
    """
    Create a graph partitioning based on hdbscan. Uses the distances between nodes to create a sparse matrix and applies
    hdbscan to the matrix to create clustering labels.

    @param graph: The graph that should be partitioned into communities.
    @return:
    """

    inverted_weights = [1 - edge_weight for edge_weight in graph.es["weight"]]
    graph.es["weight"] = inverted_weights

    distance_matrix = create_distance_matrix(graph)
    # distance_matrix = graph.get_adjacency_sparse(attribute="weight")

    clusterer = hdbscan.HDBSCAN(metric="precomputed", min_samples=2).fit(distance_matrix)
    labels = clusterer.labels_
    # print(labels)

    if min(labels) < 0:
        labels = [x + 1 for x in labels]

        palette = ig.ClusterColoringPalette(len(set(labels)))
        colors = [palette[index] for index in labels]
        graph.vs["color"] = colors

        graph.vs.select(palette[0]).delete()
        graph.simplify(multiple=True, loops=True, combine_edges=max)

    else:
        palette = ig.ClusterColoringPalette(len(set(labels)))
        colors = [palette[index] for index in labels]
        graph.vs["color"] = colors

    graph = ig.clustering.VertexClustering.FromAttribute(graph, attribute="color")

    return graph


def create_distance_matrix(graph):
    """
    Creates a distance matrix for the graph by calculating all shortest paths and
    converting them to a scipy sparse csr matrix.

    @param graph: The graph from which the matrix should be calculated.
    @return: Returns a scipy sparse csr matrix containing all edges and shortest paths, (vertex, vertex) path-weight
    """

    path_weight = []
    vertex_from_list = []
    vertex_to_list = []
    vertex_from = 0

    for vertex in graph.vs:
        list_edges_shortest_path = graph.get_shortest_paths(vertex, to=None, weights="weight", mode='out',
                                                            output="epath")
        vertex_to = 0

        for edge_list in list_edges_shortest_path:
            if edge_list:
                vertex_from_list.append(vertex_from)
                vertex_to_list.append(vertex_to)
                path_weight.append(sum(graph.es.select(edge_list)["weight"]))
            else:
                vertex_from_list.append(vertex_from)
                vertex_to_list.append(vertex_to)
                path_weight.append(0)

            vertex_to += 1
        vertex_from += 1

    distance_matrix = csr_matrix((path_weight, (vertex_from_list, vertex_to_list)))

    return distance_matrix


def merge_by_list(graph_as_clustering):
    """
    Merges the vertices by list and reformat the attributes into a single list and average for the probabilities.

    @param graph_as_clustering: An iGraph vertex clustering.
    @return: The merged graph with newly distributed attributes.
    """
    # combine strings of nodes by components and take attributes by list
    graph = graph_as_clustering.cluster_graph(combine_vertices=list, combine_edges=max)

    # assign attributes after merging by list
    for vertex in graph.vs:
        probability_df_sum = vertex["probability_df"][0]
        number_of_dfs = len(vertex["probability_df"])

        # set vertex index
        vertex_index = vertex.index

        # add elements of all probability_dfs in a vertex
        for probability_df_list in vertex["probability_df"][1:]:
            probability_df_sum = [element_list_1 + element_list_2 for element_list_1, element_list_2 in
                                  zip(probability_df_sum, probability_df_list)]

        # create new list of attributes for merged nodes
        graph.vs[vertex_index]["probability_df"] = [elements_df / number_of_dfs for elements_df in probability_df_sum]
        graph.vs[vertex_index]["name"] = sum(vertex["name"], [])
        graph.vs[vertex_index]["clustering"] = sum(vertex["clustering"], [])
        graph.vs[vertex_index]["cell"] = sum(vertex["cell"], [])
        graph.vs[vertex_index]["level"] = max(vertex["level"])
        graph.vs[vertex_index]["cell_index"] = vertex["cell_index"][0]

    return graph


def jaccard_index_two_vertices(vertex_1, vertex_2):
    """
    Calculates the jaccard index based on the included cells in a vertex.

    @param vertex_1: iGraph vertex object. Needs attribute cell; vertex_1["cell"]
    @param vertex_2: iGraph vertex object. Needs attribute cell; vertex_1["cell"]
    @return: The calculated jaccard index.
    """

    # creates sets based on the vertex attribute cell
    set_1 = set(sum(vertex_1["cell"], []))
    set_2 = set(sum(vertex_2["cell"], []))

    # intersection and union for the jaccard index
    intersection = set_1.intersection(set_2)
    union = set_1.union(set_2)

    # calculate the jaccard index
    jaccard = len(intersection) / len(union)

    return jaccard


def weighted_jaccard(probability_node_1, probability_node_2):
    """
    Weighted jaccard-index based on the paper "Finding the Jaccard Median"; http://theory.stanford.edu/~sergei/papers/soda10-jaccard.pdf

    @param probability_node_1: List of probabilities (vertex_1) created from the cells occurring in the meta node,
        created with graph_nodes_cells_to_df and split up by column and turn int list with values.
    @param probability_node_2:  List of probabilities (vertex_2) created from the cells occurring in the meta node,
        created with graph_nodes_cells_to_df and split up by column and turn int list with values.
    @return: The weighted probability_node_2 index; int.
    """

    sum_min = sum([min(compare_elements) for compare_elements in zip(probability_node_1, probability_node_2)])
    sum_max = sum([max(compare_elements) for compare_elements in zip(probability_node_1, probability_node_2)])

    if sum_max == 0:
        weighted_jaccard_index = 0
    else:
        weighted_jaccard_index = sum_min / sum_max

    return weighted_jaccard_index
