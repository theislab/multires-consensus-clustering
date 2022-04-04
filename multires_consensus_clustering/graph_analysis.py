import igraph as ig
import numpy as np
import multires_consensus_clustering as mcc
from scipy.sparse import csr_matrix
from numba import njit


def merge_by_list(graph_as_clustering):
    """
    Merges the vertices by list and reformat the attributes into a single list and average for the probabilities.

    @param graph_as_clustering: An iGraph vertex clustering.
    @return: The merged graph with newly distributed attributes.
    """
    # combine strings of nodes by components and take attributes by list
    graph = graph_as_clustering.cluster_graph(combine_vertices=list, combine_edges=max)

    # if done in the multi-res-graph step the edge attributes would be list of list and have to bel flattened
    if type(graph.vs[0]["name"][0]) == list:
        # assign attributes after merging by list
        for vertex in graph.vs:
            # set vertex index
            vertex_index = vertex.index

            # create new list of attributes for merged nodes
            graph.vs[vertex_index]["name"] = sum(vertex["name"], [])
            graph.vs[vertex_index]["clustering"] = sum(vertex["clustering"], [])
            graph.vs[vertex_index]["cell"] = sum(vertex["cell"], [])
            graph.vs[vertex_index]["level"] = max(vertex["level"])
            graph.vs[vertex_index]["cell_index"] = vertex["cell_index"][0]

    return graph


def create_distance_matrix(graph):
    """
    Creates a distance matrix for the graph by calculating all shortest paths and
    converting them to a scipy sparse csr matrix.

    @param graph: The graph from which the matrix should be calculated.
    @return: Returns a scipy sparse csr matrix containing all edges and shortest paths, (vertex, vertex) path-weight
    """

    # create variables
    path_weight, vertex_from_list, vertex_to_list, vertex_from = [], [], [], 0

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


@njit
def weighted_jaccard(probability_node_1, probability_node_2):
    """
    Weighted jaccard-index based on the paper "Finding the Jaccard Median"; http://theory.stanford.edu/~sergei/papers/soda10-jaccard.pdf

    @param probability_node_1: List of probabilities (vertex_1) created from the cells occurring in the meta node,
        created with graph_nodes_cells_to_df and split up by column and turn int list with values.
    @param probability_node_2:  List of probabilities (vertex_2) created from the cells occurring in the meta node,
        created with graph_nodes_cells_to_df and split up by column and turn int list with values.
    @return: The weighted probability_node_2 index; int.
    """
    sum_min = 0
    sum_max = 0
    for probability_index in range(len(probability_node_1)):
        sum_min += min(probability_node_1[probability_index], probability_node_2[probability_index])
        sum_max += max(probability_node_1[probability_index], probability_node_2[probability_index])

    if sum_max == 0:
        weighted_jaccard_index = 0
    else:
        weighted_jaccard_index = sum_min / sum_max

    return weighted_jaccard_index
