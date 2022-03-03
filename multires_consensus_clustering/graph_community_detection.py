import igraph as ig
import numpy as np
import multires_consensus_clustering as mcc
import hdbscan


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
        vertex_clustering = ig.Graph.community_fastgreedy(G, weights="weight").as_clustering()

    elif detection_algorithm == "newman2006":
        # newman2006 graph community detection, more community then the others
        vertex_clustering = ig.Graph.community_leading_eigenvector(G, weights="weight")

    elif detection_algorithm == "louvain":
        # louvain methode for graph community detection
        vertex_clustering = ig.Graph.community_multilevel(G, weights="weight")

    elif detection_algorithm == "leiden":
        # leiden methode for graph community detection, improvement of the louvain methode
        # https://arxiv.org/abs/1810.08473
        vertex_clustering = ig.Graph.community_leiden(G, weights="weight", objective_function="modularity", n_iterations=-1)

    else:
        # if none of the above are selected -> automatically uses leiden
        vertex_clustering = ig.Graph.community_leiden(G, weights="weight", objective_function="modularity", n_iterations=-1)

    return vertex_clustering


def hdbscan_community_detection(graph):
    """
    Create a graph partitioning based on HDBscan. Uses the distances between nodes to create a sparse matrix and applies
    HDBscan to the matrix to create clustering labels.

    @param graph: The graph that should be partitioned into communities.
    @return:
    """

    # invert edge weights for HDBscan as closer in the graph means close to 1 and in HDBscan close to 0.
    inverted_weights = [1 - edge_weight for edge_weight in graph.es["weight"]]
    graph.es["weight"] = inverted_weights

    # check if distances are not infinite
    if graph.average_path_length(directed=False, unconn=False) != np.inf:
        # create a distance matrix for the graph if possible
        distance_matrix = mcc.create_distance_matrix(graph)

        clusterer = hdbscan.HDBSCAN(metric="precomputed", min_samples=2).fit(distance_matrix)
        labels = clusterer.labels_

        # hdbscan has an integrated outlier detection for clustering -> delete those nodes
        if min(labels) < 0:
            labels = [x + 1 for x in labels]

            palette = ig.ClusterColoringPalette(len(set(labels)))
            colors = [palette[index] for index in labels]
            graph.vs["color"] = colors

            graph.vs.select(palette[0]).delete()
            graph.simplify(multiple=True, loops=True, combine_edges=max)

        # if no outliers cluster the graph and create a vertex sequence
        else:
            palette = ig.ClusterColoringPalette(len(set(labels)))
            colors = [palette[index] for index in labels]
            graph.vs["color"] = colors

        # create vertex sequence based on hdbscan
        vertex_clustering = ig.clustering.VertexClustering.FromAttribute(graph, attribute="color")

    # if infinite distances catches case by running leiden community detection.
    else:
        print("Graph has distances np.inf, thus HDBscan won't compute.")
        vertex_clustering = igraph_community_detection(graph, "leiden")

    return vertex_clustering


def component_merger(graph, threshold_edges_to_delete):
    """
    If the graph is split into different components merges the components and takes the average of the attributes
    or the first attribute.

    @param threshold_edges_to_delete: Integer between 0 and 1 to te the limit on what edge weights the edges are deleted.
    @param graph: The graph on which the function works, iGraph graph.
    @return: The clustering based on the components, iGraph VertexClustering.
    """

    # delete all edges below the selected threshold
    graph = mcc.delete_edges_below_threshold(graph, threshold_edges_to_delete, delete_single_nodes=False)

    # merge components
    graph_component = graph.clusters()

    return graph_component


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
