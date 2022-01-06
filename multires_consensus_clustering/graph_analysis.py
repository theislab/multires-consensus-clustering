import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from multires_consensus_clustering import meta_graph as mg
import itertools
import multires_consensus_clustering as mcc
import seaborn as sns
import hdbscan
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import itertools
import networkx as nx


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

    elif detection_algorithm == "all":
        # return list with vertex_clustering for all algorithms
        graph_list = []

        # fast_greedy
        graph_list.append(ig.Graph.community_fastgreedy(G, weights="weight").as_clustering())

        # infomap Martin Rosvall and Carl T. Bergstrom.
        graph_list.append(ig.Graph.community_infomap(G, edge_weights="weight"))

        # label propagation method of Raghavan et al.
        #graph_list.append(ig.Graph.community_label_propagation(G, weights="weight"))

        # newman2006
        graph_list.append(ig.Graph.community_leading_eigenvector(G, weights="weight"))

        # louvain, of Blondel et al.
        graph_list.append(ig.Graph.community_multilevel(G, weights="weight"))

        # betweenness of the edges in the network.
        #graph_list.append(ig.Graph.community_edge_betweenness(G, weights="weight").as_clustering())

        # spinglass community detection method of Reichardt & Bornholdt.
        #graph_list.append(ig.Graph.community_spinglass(G, weights="weight"))

        # detection algorithm of Latapy & Pons, based on random walks.
        #graph_list.append(ig.Graph.community_walktrap(G, weights="weight").as_clustering())

        # community structure of the graph using the Leiden algorithm of Traag, van Eck & Waltman, to many clusters.
        #graph = ig.Graph.community_leiden(G, weights="weight")

        #ig.plot(graph)
        #graph_list.append(ig.Graph.community_leiden(G, weights="weight"))
        #ig.plot(graph_list[0])
        #ig.plot(graph_list[1])
        #ig.plot(graph_list[2])
        return graph_list


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
        range_edges = range(number_edges)
        edge_weights = graph.es["weight"]
        mean_edge_value = sum(edge_weights) / len(edge_weights)
        mean_edge_value_list = [mean_edge_value] * len(edge_weights)

        # edge weights sorted by edge id
        """
        plt.bar(range_edges, edge_weights)
        plt.plot(range_edges, mean_edge_value_list, linestyle='--', color='red')
        plt.show()
        """

        if plot_on_off:
            # edge weights sorted from high to low; 1 -> 0
            plt.bar(range_edges, sorted(edge_weights, reverse=True))
            plt.plot(range_edges, mean_edge_value_list, linestyle='--', color='red')
            plt.show()

        return mean_edge_value


def intersect_two_graphs_lists(graph_list_1, graph_list_2):
    """
    Finds the intersection graph of two graphs, for every combination of graphs in the two given lists.
    @param graph_list_1: List with graphs, iGraph graphs.
    @param graph_list_2: List with graphs, iGraph graphs.
    @return: Returns a list with all graph intersections of the two lists.
    """

    intersection_list = []
    for subgraph_0 in graph_list_1:
        for subgraph_1 in graph_list_2:
            intersection = ig.intersection([subgraph_0,subgraph_1], keep_all_vertices=False, byname="auto")
            intersection_list.append(intersection)

    return intersection_list


def consensus_graph(graph):
    """
    Uses multiple graph community detection algorithms to find all possible communities. Uses graph intersection to
    dived the graph into the smallest partition of the graphs combined.
    @param graph: The graph which should be partitioned into multiple subgraphs/communities.
    @return: A union of those subgraphs, not connected.
    """

    list_graphs = igraph_community_detection(graph, detection_algorithm="all")
    graph_community_list = []

    for graphs in list_graphs:
        subgraph_clustering = graphs.subgraphs()
        graph_community_list.append(subgraph_clustering)

    intersection_list = graph_community_list[0]
    for subgraph_clustering in graph_community_list[1:]:
        intersection_list = intersect_two_graphs_lists(intersection_list, subgraph_clustering)

    cluster_index = 0
    for intersection in intersection_list:
        if intersection:
            for vertex in intersection.vs:
                vertex_index = graph.vs.find(name=vertex["name"]).index
                graph.vs[vertex_index]["color"] = cluster_index
        cluster_index += 1

    graph = ig.clustering.VertexClustering.FromAttribute(graph, attribute="color")

    return graph


def hdbscan_community_detection(graph):
    """
    Create a graph partitioning based on hdbscan. Uses the distances between nodes to create a sparse matrix and applies
    hdbscan to the matrix to create clustering labels.

    @param graph: The graph that should be partitioned into communities.
    @return:
    """

    """
    combinations_of_vertices = list(itertools.combinations(range(graph.vcount()), 2))
    added_edge_weight = np.inf
    for edge in combinations_of_vertices:
            if edge not in graph.get_edgelist():
                graph.add_edges([edge])
                edge_index = graph.get_eid(edge[0], edge[1])
                graph.es[edge_index]["weight"] = added_edge_weight
    """

    path_weight = []
    vertex_from_list = []
    vertex_to_list = []
    vertex_from = 0

    inverted_weights = [1 - edge_weight for edge_weight in graph.es["weight"]]
    graph.es["weight"] = inverted_weights

    for vertex in graph.vs:
        list_edges_shortest_path = graph.get_shortest_paths(vertex["name"], to=None, weights="weight", mode='out', output="epath")
        vertex_to = 0

        for edge_list in list_edges_shortest_path:
            if edge_list:
                vertex_from_list.append(vertex_from)
                vertex_to_list.append(vertex_to)
                path_weight.append(sum(graph.es.select(edge_list)["weight"]))

            vertex_to += 1
        vertex_from += 1

    distance_matrix = csr_matrix((path_weight, (vertex_from_list, vertex_to_list)), shape=(len(path_weight), len(path_weight)))

    clusterer = hdbscan.HDBSCAN(metric="precomputed").fit(distance_matrix)
    labels = clusterer.labels_

    if min(labels) < 0:
        labels = [x+1 for x in labels]

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

    #ig.plot(graph)

    return graph

