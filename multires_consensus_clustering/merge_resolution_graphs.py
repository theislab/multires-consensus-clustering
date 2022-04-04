import numpy as np
import pandas as pd
import igraph as ig
import multires_consensus_clustering as mcc
import time


def multiresolution_graph(clustering_data, settings_data, list_resolutions, neighbour_based):
    """
    Creates a multi-resolution graph based on the resolutions given in the list_resolutions.
    Can either create a graph where all resolution vertices are connected or only the neighbouring resolutions are connected.

    @param settings_data: Settings data about the clusters.
    @param clustering_data: The cluster data.
    @param neighbour_based: Boolean to decided on the way to connect the vertices across resolutions.
    @param list_resolutions: The list containing the different resolutions, e.g. [3,5,9,20, ... ] or "all"
    @return: The mulit-graph as a iGraph graph.
    """

    # set maximum level for hierarchy plot
    len_list_resolutions = len(list_resolutions)
    level_count = len_list_resolutions + 1

    # check if list resolution contains less the two resolutions
    if len_list_resolutions <= 1:
        print("More then one resolution needed for multi resolution graph.")

        return mcc.meta_graph(clustering_data, settings_data, list_resolutions[0])

    # create the multi graph
    else:
        if list_resolutions == "all":
            bins_clusterings = mcc.bin_n_clusters(settings_data["n_clusters"])
            list_resolutions = [int(first_number_clusters[0]) for first_number_clusters in bins_clusterings]
        else:
            # sort resolution list in cases not sorted
            list_resolutions.sort()

        # create first graph and assign the level
        resolution_1 = mcc.meta_graph(clustering_data, settings_data, list_resolutions[0])
        resolution_1.vs["level"] = [level_count] * resolution_1.vcount()

        # create new attribute to save the cell probabilities in a meta node
        probability_df = mcc.graph_nodes_cells_to_df(resolution_1, clustering_data)
        resolution_1.vs["probability_df"] = [probability_df[column].values for column in probability_df.columns]

        resolution_1.vs[0]["cell_index"] = probability_df.index.tolist()

        # delete all edges of the old graph
        mcc.delete_edges_single_resolution(resolution_1)

        # change level count
        level_resolution_2 = level_count - 1

        # select all resolutions except the first
        list_resolutions = list_resolutions[1:]

        # create multi-graph using the rest of the list_resolutions
        for resolution in list_resolutions:

            # create graph and assign the level
            resolution_2 = mcc.meta_graph(clustering_data, settings_data, resolution)
            resolution_2.vs["level"] = [level_resolution_2] * resolution_2.vcount()

            # delete all edges of the old graph
            mcc.delete_edges_single_resolution(resolution_2)

            # create multi graph based on neighbours or connect all vertices
            if neighbour_based:
                resolution_1 = mcc.merge_two_resolution_graphs(resolution_1, resolution_2,
                                                               current_level=level_resolution_2 + 1,
                                                               neighbours=True, clustering_data=clustering_data)
            else:
                # connect all vertices
                resolution_1 = mcc.merge_two_resolution_graphs(resolution_1, resolution_2, current_level=None,
                                                               neighbours=False, clustering_data=clustering_data)

            # set level for next graph
            level_resolution_2 -= 1

        # return the final multi-graph
        return resolution_1


def delete_edges_single_resolution(graph):
    """
    Deletes all edges with an edge weight above 0 -> these are all edges of the graph.
    @param graph: The given graph, iGraph graph.
    @return: The graph without edges; only vertices with attributes
    """

    return graph.delete_edges()


def merge_two_resolution_graphs(graph_1, graph_2, current_level, neighbours, clustering_data):
    """
    Merges two graphs;
    Either connects all vertices or only the bins neighbouring each other. All edges are based on the jaccard-index.

    @param clustering_data: The clustering data from which the meta graphs are build.
    @param neighbours: Connects all vertices or only neighbouring resolutions; Boolean
    @param current_level: The level of the last added vertices.
        Need so the new vertices are only connected to the latest vertices and not all.
    @param graph_1: The first graph, iGraph graph.
    @param graph_2: The second graph, iGraph graph.
    @return: The merged graph.
    """

    # create edge lists
    edge_list, edge_weights = [], []

    # to check the graph merger visually
    graph_1.vs["graph"] = [1] * graph_1.vcount()
    graph_2.vs["graph"] = [2] * graph_2.vcount()

    # add cell probability to the second graph with is added in the new layer (level)
    probability_df = mcc.graph_nodes_cells_to_df(graph_2, clustering_data)
    graph_2.vs["probability_df"] = [probability_df[column].values for column in probability_df.columns]

    # creates a graph based on the two given resolutions
    graph = graph_1.disjoint_union(graph_2)

    # connects vertices based on neighbouring resolutions.
    if neighbours:
        for vertex_1 in graph.vs.select(graph=1):
            for vertex_2 in graph.vs.select(graph=2):
                # connects only bins next to each other
                if vertex_1["level"] == current_level and vertex_2["level"] == current_level - 1:
                    # calculate edge weight
                    edge_weight = mcc.weighted_jaccard(vertex_1["probability_df"], vertex_2["probability_df"])

                    # if the edge_weight is greater 0 the edge is added
                    if edge_weight != 0:
                        edge_list.append((vertex_1, vertex_2))
                        edge_weights.append(edge_weight)

    # connects all vertices
    else:
        for vertex_1 in graph.vs.select(graph=1):
            for vertex_2 in graph.vs.select(graph=2):
                # calculate edge weight
                # edge_weight = mcc.jaccard_index_two_vertices(vertex_1, vertex_2)
                edge_weight = mcc.weighted_jaccard(vertex_1["probability_df"], vertex_2["probability_df"])

                # if the edge_weight is greater 0 the edge is added
                if edge_weight != 0:
                    edge_list.append((vertex_1, vertex_2))
                    edge_weights.append(edge_weight)

    # add edges to the graph
    graph.add_edges(edge_list)

    # add edge weights to the graph
    graph.es["weight"] = edge_weights

    return graph


def reconnect_graph(graph):
    """
    Reconnects the graph. Useful if after merging the graph is just a set of separated nodes.

    @param graph: The graph, in the ideal case a set of discrete nodes.
    @return: The connected graph. Edges are chosen that in the end the graph has a tree structure,
        based on the level of the nodes.
    """

    # check if graph has more than one level
    if len(set(graph.vs["level"])) != 1:
        # if the graph has more then one level a tree structure can be created, using the resolution of the graphs
        for vertex_1 in graph.vs:
            level_vertex_1 = vertex_1["level"]
            for vertex_2 in graph.vs:
                if vertex_1 != vertex_2:
                    # connects only bins next to each other
                    if vertex_2["level"] < level_vertex_1:
                        # calculate edge weight
                        edge_weight = mcc.weighted_jaccard(vertex_1["probability_df"], vertex_2["probability_df"])

                        # if the edge_weight is greater 0 the edge is added
                        if edge_weight != 0:
                            index_1 = vertex_1.index
                            index_2 = vertex_2.index
                            graph.add_edge(index_1, index_2, weight=edge_weight)

                        # check if better edge is available for tree structure
                        edge_list_vertex_2 = vertex_2.all_edges()

                        # if the vertex has only one edge no better is available
                        if len(edge_list_vertex_2) > 1:
                            edges_to_delete = []
                            current_best_weight = 0
                            current_best_edge = None

                            for edge in edge_list_vertex_2:
                                edges_to_delete.append(edge)
                                edge_weight = edge["weight"]

                                # only check edge going from a lower resolution to a higher resolution -> tree structure
                                if graph.vs[edge.target]["level"] < vertex_2["level"]:
                                    edges_to_delete.pop(len(edges_to_delete) - 1)
                                else:
                                    # choose the edge with the best edge weight
                                    if edge_weight > current_best_weight:
                                        edges_to_delete.pop(len(edges_to_delete) - 1)
                                        if current_best_edge is not None:
                                            edges_to_delete.append(current_best_edge)
                                        current_best_weight = edge_weight
                                        current_best_edge = edge

                                    # if the edges have the same edge weight choose the higher resolution one
                                    elif edge_weight == current_best_weight:
                                        last_edge = edges_to_delete.pop(len(edges_to_delete) - 1)
                                        if graph.vs[last_edge.target]["level"] > graph.vs[edge.target]["level"]:
                                            edges_to_delete.append(last_edge)
                                        else:
                                            edges_to_delete.append(edge)

                            graph.delete_edges(edges_to_delete)

    # otherwise returns the maximum spanning tree
    else:
        # calculate all edge weights for every vertex in the graph
        for vertex_1 in graph.vs:
            for vertex_2 in graph.vs:
                if vertex_1 != vertex_2:
                    # calculate edge weight
                    edge_weight = mcc.weighted_jaccard(vertex_1["probability_df"], vertex_2["probability_df"])

                    # if the edge_weight is greater 0 the edge is added
                    if edge_weight != 0:
                        index_1 = vertex_1.index
                        index_2 = vertex_2.index
                        graph.add_edge(index_1, index_2, weight=edge_weight)

        # invert edge weights
        inverted_weights = [1 - edge_weight for edge_weight in graph.es["weight"]]

        # create the minimum spanning tree using the inverted edge weight -> maximum spanning tree
        graph = graph.spanning_tree(weights=inverted_weights, return_tree=True)

    return graph


def multires_community_detection(graph, clustering_data, community_detection, merge_edges_threshold, outlier_detection,
                                 outlier_detection_threshold):
    """
    Uses louvain community detection on the multi-resolution graph and creates a clustering tree with reconnect_graph.
    Optional clean up of the clustering tree with edge merging.

    @param outlier_detection_threshold: Value on which the outlier detection is based, can be 0 to 1.
    @param clustering_data: The clustering data on which the graph is based
    @param outlier_detection: String: "probability" or "hdbscan" to choose on which the outlier detciton is based.
    @param merge_edges_threshold: Threshold for edges to merge at the end, should probably be between 0.8-1.
    @param community_detection: "leiden", "hdbscan", "component" or else automatically louvain,
        detects community with the named algorithms
    @param graph: The mulit resolution graph, iGraph graph.
    @return: A clustering tree, igraph Graph.
    """

    # choose outlier detection
    if outlier_detection == "probability":
        graph = mcc.filter_by_node_probability(graph, threshold=outlier_detection_threshold)
    elif outlier_detection == "hdbscan":
        graph = mcc.hdbscan_outlier(graph, threshold=1 - outlier_detection_threshold, plot_on_off=False)

    # community detection
    if community_detection == "leiden":
        # uses the leiden algorithm for community detection
        vertex_clustering = ig.Graph.community_leiden(graph, weights="weight")
    elif community_detection == "hdbscan" and graph.vcount() > 1 and graph.ecount() > 0:
        # use hdbscan for community detection
        vertex_clustering = mcc.hdbscan_community_detection(graph)
    elif community_detection == "component":
        # splits the graph into components and merges these components into a single node
        vertex_clustering = mcc.component_merger(graph, threshold_edges_to_delete=0.99)
    else:
        # if nothing is selected uses louvain community detection
        vertex_clustering = ig.Graph.community_multilevel(graph, weights="weight")

    # combines attributes to a list
    graph = mcc.merge_by_list(vertex_clustering)

    # recalculate the probabilities based on the new merged nodes
    new_probabilities = mcc.graph_nodes_cells_to_df(graph, clustering_data)
    graph.vs["probability_df"] = [new_probabilities[column].values for column in new_probabilities.columns]

    # delete edges and create graph tree structure if there is more than one node
    if graph.vcount() != 1:
        graph.delete_edges()
        graph = reconnect_graph(graph)

        if graph.ecount() > 0:
            # clean up graph by merging some edges
            graph = mcc.merge_edges_weight_above_threshold(graph, threshold=merge_edges_threshold)

            # recalculate the probabilities based on the new merged nodes
            new_probabilities = mcc.graph_nodes_cells_to_df(graph, clustering_data)
            graph.vs["probability_df"] = [new_probabilities[column].values for column in new_probabilities.columns]

            if graph.vcount() != 1:
                graph.delete_edges()
                graph = reconnect_graph(graph)
    return graph
