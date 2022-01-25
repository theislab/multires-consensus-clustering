import multires_consensus_clustering as mcc
import numpy as np
import pandas as pd
import igraph as ig


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
        for vertex_1 in graph_1.vs:
            for vertex_2 in graph_2.vs:
                # connects only bins next to each other
                if vertex_1["level"] == current_level:
                    # edge_weight = jaccard_index_two_vertices(vertex_1, vertex_2)
                    edge_weight = mcc.weighted_jaccard(vertex_1["probability_df"], vertex_2["probability_df"])

                    # if the edge_weight is greater 0 the edge is added
                    if edge_weight != 0:
                        index_1 = graph.vs.find(name=vertex_1["name"]).index
                        index_2 = graph.vs.find(name=vertex_2["name"]).index
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
