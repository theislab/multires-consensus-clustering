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


def merge_two_resolution_graphs(graph_1, graph_2, current_level, neighbours, clustering_data):
    """
    Merges two graphs;
    Either connects all vertices or only the bins neighbouring each other. All edges are based on the jaccard-index.

    @param clustering_data:
    @param neighbours: Connects all vertices or only neighbouring resolutions; Boolean
    @param current_level: The level of the last added vertices.
        Need so the new vertices are only connected to the latest vertices and not all.
    @param graph_1: The first graph, iGraph graph.
    @param graph_2: The second graph, iGraph graph.
    @return: The merged graph.
    """

    """
    # to check the graph merger visually
    graph_1.vs["graph"] = [1] * graph_1.vcount()
    graph_2.vs["graph"] = [2] * graph_2.vcount()
    """

    # add cell probability to the second graph with is added in the new layer (level)
    probability_df = mcc.graph_nodes_cells_to_df(graph_2, clustering_data)
    graph_2.vs["probability_df"] = [probability_df[column].values for column in probability_df.columns]

    # creates a graph based on the two given resolutions
    graph = graph_1.disjoint_union(graph_2)

    # connects vertices based on neighbouring resolutions.
    if neighbours:
        for vertex_1 in graph.vs:
            for vertex_2 in graph.vs:
                # connects only bins next to each other
                if vertex_1["level"] == current_level and vertex_2["level"] == current_level - 1:
                    # edge_weight = jaccard_index_two_vertices(vertex_1, vertex_2)
                    edge_weight = mcc.weighted_jaccard(vertex_1["probability_df"], vertex_2["probability_df"])

                    # if the edge_weight is greater 0 the edge is added
                    if edge_weight != 0:
                        index_1 = vertex_1.index
                        index_2 = vertex_2.index
                        graph.add_edge(index_1, index_2, weight=edge_weight)

    # connects all vertices
    else:
        for vertex_1 in graph_1.vs:
            for vertex_2 in graph_2.vs:
                # edge_weight = jaccard_index_two_vertices(vertex_1, vertex_2)
                edge_weight = mcc.weighted_jaccard(vertex_1["probability_df"], vertex_2["probability_df"])

                # if the edge_weight is greater 0 the edge is added
                if edge_weight != 0:
                    index_1 = graph.vs.find(name=vertex_1["name"]).index
                    index_2 = graph.vs.find(name=vertex_2["name"]).index
                    graph.add_edge(index_1, index_2, weight=edge_weight)

    """
    # to check the graph merger visually 
    palette = ig.ClusterColoringPalette(2)
    colors = [palette[index - 1] for index in graph.vs["graph"]]
    graph.vs["color"] = colors
    
    ig.plot(graph)
    """

    # merge all edges with edge weight == 1
    # graph = mcc.merge_edges_weight_1(graph)

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

                    edge_weight = jaccard_index_two_vertices(vertex_1, vertex_2)
                    # edge_weight = mcc.weighted_jaccard(vertex_1["probability_df"], vertex_2["probability_df"])

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
                        lowest_resolution = np.inf

                        for edge in edge_list_vertex_2:
                            edges_to_delete.append(edge)
                            edge_weight = edge["weight"]

                            if graph.vs[edge.target]["level"] < vertex_2["level"]:
                                edges_to_delete.pop(len(edges_to_delete) - 1)
                            else:
                                if edge_weight > current_best_weight:
                                    edges_to_delete.pop(len(edges_to_delete) - 1)
                                    if current_best_edge is not None:
                                        edges_to_delete.append(current_best_edge)
                                    current_best_weight = edge_weight
                                    current_best_edge = edge

                        graph.delete_edges(edges_to_delete)

    return graph


def component_merger(graph, reconnect_graph, combine_by):
    """
    If the graph is split into different components merges the components and takes the average of the attributes
    or the first attribute.

    @param combine_by: Comines by "list" or first node attribute otherwise.
    @param reconnect_graph: True or False parameter, reconnects the components.
    @param graph: The graph on which the function works, iGraph graph.
    @return: The merged graph, iGraph graph.
    """

    # merge components
    graph_component = graph.clusters()

    if combine_by == "list":
        # combine strings of nodes by components and take attributes by list
        graph = graph_component.cluster_graph(combine_vertices=list, combine_edges=max)

        # assign attributes after merging by list
        for vertex in graph.vs:
            probability_df_sum = vertex["probability_df"][0]
            number_of_dfs = len(vertex["probability_df"])

            # add elements of all probability_dfs in a vertex
            for probability_df_list in vertex["probability_df"][1:]:
                probability_df_sum = [element_list_1 + element_list_2 for element_list_1, element_list_2 in
                                      zip(probability_df_sum, probability_df_list)]

            # create new list of attributes for merged nodes
            vertex["probability_df"] = [elements_df / number_of_dfs for elements_df in probability_df_sum]
            vertex["name"] = sum(vertex["name"], [])
            vertex["clustering"] = sum(vertex["clustering"], [])
            vertex["cell"] = sum(vertex["cell"], [])
            vertex["level"] = max(vertex["level"])
            vertex["cell_index"] = vertex["cell_index"][0]

    else:
        # combine strings of nodes by components and take attributes from the first node
        graph = graph_component.cluster_graph(combine_vertices="first", combine_edges=max)

    # reconnect the components
    if reconnect_graph:
        graph = reconnect_graph(graph)

    return graph


def multires_community_detection(graph):
    """
    Uses louvain community detection on the multi-resolution graph and creates a clustering tree with reconnect_graph.
    Optional clean up of the clustering tree with edge merging.

    @param graph: The mulit resolution graph, iGraph graph.
    @return: A clustering tree, igraph Graph.
    """
    # community detection
    graph = ig.Graph.community_multilevel(graph, weights="weight").cluster_graph(combine_vertices="first",
                                                                                 combine_edges=max)

    # delete edges and create graph tree structure
    graph.es.select(weight_gt=0).delete()
    graph = reconnect_graph(graph)

    # clean up graph by merging some edges
    #graph = mcc.merge_edges_weight_above_threshold(graph, threshold=0.9)

    for vertex in graph.vs:
        print(len(vertex["probability_df"]))

    return graph
