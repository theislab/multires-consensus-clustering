import numpy as np
import igraph as ig


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


def delete_one_node_communities(vertex_clustering):
    """

    @param graph:
    @return:
    """
    subgraph_list = vertex_clustering.subgraphs()
    sum_subgraphs = sum([subgraph.vcount() for subgraph in subgraph_list])
    normalized_subgraph_size = sum_subgraphs / len(subgraph_list)

    #print(vertex_clustering.graph.vcount())

    for subgraph in subgraph_list:
        if subgraph.vcount() < normalized_subgraph_size:
            vertex_clustering.graph.vs.select(name_in=subgraph.vs["name"]).delete()

    #print(vertex_clustering.graph.vcount())

    return vertex_clustering.graph
