import time

import multires_consensus_clustering as mcc
import numpy as np
import pandas as pd
import igraph as ig


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
    level_count = len(list_resolutions) + 1

    # check if list resolution contains less the two resolutions
    if len(list_resolutions) <= 1:
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

        # create multi-graph using the rest of the list_resolutions
        for resolution in list_resolutions[1:]:

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

    return graph.es.select(weight_gt=0).delete()


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
                        index_1 = vertex_1.index
                        index_2 = vertex_2.index
                        graph.add_edge(index_1, index_2, weight=edge_weight)

    # connects all vertices
    else:
        for vertex_1 in graph.vs.select(graph=1):
            for vertex_2 in graph.vs.select(graph=2):
                # calculate edge weight
                edge_weight = mcc.weighted_jaccard(vertex_1["probability_df"], vertex_2["probability_df"])

                # if the edge_weight is greater 0 the edge is added
                if edge_weight != 0:
                    index_1 = vertex_1.index
                    index_2 = vertex_2.index
                    graph.add_edge(index_1, index_2, weight=edge_weight)

    return graph


def reconnect_graph(graph):
    """
    Reconnects the graph. Useful if after merging the graph is just a set of separated nodes.

    @param graph: The graph, in the ideal case a set of discreate nodes.
    @return: The connected graph. Edges are chosen that in the end the graph has a tree structure,
        based on the level of the nodes.
    """

    for vertex_1 in graph.vs:
        level_vertex_1 = vertex_1["level"]
        for vertex_2 in graph.vs:
            if vertex_1 != vertex_2:
                # connects only bins next to each other
                if vertex_2["level"] < level_vertex_1:
                    # calculate edge weight
                    # edge_weight = mcc.jaccard_index_two_vertices(vertex_1, vertex_2)
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

    return graph


def component_merger(graph, reconnect_graph_true_false, combine_by, threshold_edges_to_delete):
    """
    If the graph is split into different components merges the components and takes the average of the attributes
    or the first attribute.

    @param threshold_edges_to_delete: Integer between 0 and 1 to te the limit on what edge weights the edges are deleted.
    @param combine_by: Comines by "list" or first node attribute otherwise.
    @param reconnect_graph_true_false: True or False parameter, reconnects the components.
    @param graph: The graph on which the function works, iGraph graph.
    @return: The merged graph, iGraph graph.
    """

    graph = mcc.delete_edges_below_threshold(graph, 0.8, delete_single_nodes=False)

    # merge components
    graph_component = graph.clusters()

    if combine_by == "list":
        # uses louvain community detection to merge the graph and combines attributes to a list/average of probabilities
        graph = mcc.merge_by_list_louvain(graph)

    else:
        # combine strings of nodes by components and take attributes from the first node
        graph = graph_component.cluster_graph(combine_vertices="first", combine_edges=max)

    # reconnect the components
    if reconnect_graph_true_false:
        graph = reconnect_graph(graph)

    return graph


def multires_community_detection(graph, combine_by):
    """
    Uses louvain community detection on the multi-resolution graph and creates a clustering tree with reconnect_graph.
    Optional clean up of the clustering tree with edge merging.

    @param combine_by: "list" or "first" combines attributes either by list or by first
    @param graph: The mulit resolution graph, iGraph graph.
    @return: A clustering tree, igraph Graph.
    """
    # community detection
    if combine_by == "list":
        # uses leiden community detection to merge the graph and combines attributes to a list/average of probabilities
        graph = mcc.hdbscan_community_detection(graph)
        graph = mcc.merge_by_list(graph)

    else:
        # combine strings of nodes by components and take attributes from the first node
        graph = mcc.hdbscan_community_detection(graph).cluster_graph(combine_vertices="first", combine_edges=max)

    # delete edges and create graph tree structure
    graph.es.select(weight_gt=0).delete()
    graph = reconnect_graph(graph)

    # clean up graph by merging some edges
    graph = mcc.merge_edges_weight_above_threshold(graph, threshold=1)

    return graph