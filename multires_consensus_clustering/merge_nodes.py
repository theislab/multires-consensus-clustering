import igraph as ig
import numpy as np
from multires_consensus_clustering import Meta_Graph as mg


def merge_nodes(graph, node_1_indices, node_2_indices):
    """
    Merges two nodes in the given graph, using the node indices.
    Combines all attributes as a list of the attributes given, e.g. graph.vs["name"] = [[C001:1,C002:2],C003:1,...]

    :param node_2_indices: Indices of node_1, type: int e.g. 1,2,3,...
    :param node_1_indices: Indices of node_2, type: int e.g. 4,5,6,...
    :param graph: The graph given, type iGraph graph object.
    :return: The graph with merged nodes.
    """
    vertex_indices = range(graph.vcount())

    # create list with all names of node_1 and node_2
    attributes_merged_nodes_name = []
    if type(graph.vs[node_1_indices]["name"]) == list:
        for names in graph.vs[node_1_indices]["name"]:
            attributes_merged_nodes_name.append(names)
    else:
        attributes_merged_nodes_name.append(graph.vs[node_1_indices]["name"])

    if type(graph.vs[node_2_indices]["name"]) == list:
        for names in graph.vs[node_2_indices]["name"]:
            attributes_merged_nodes_name.append(names)
    else:
        attributes_merged_nodes_name.append(graph.vs[node_2_indices]["name"])

    # create list with all clusterings names for node_1 and node_2
    attributes_merged_nodes_clustering = []
    if type(graph.vs[node_1_indices]["clustering"]) == list:
        for clustering in graph.vs[node_1_indices]["clustering"]:
            attributes_merged_nodes_clustering.append(clustering)
    else:
        attributes_merged_nodes_clustering.append(graph.vs[node_1_indices]["clustering"])

    if type(graph.vs[node_2_indices]["clustering"]) == list:
        for clustering in graph.vs[node_2_indices]["clustering"]:
            attributes_merged_nodes_clustering.append(clustering)
    else:
        attributes_merged_nodes_clustering.append(graph.vs[node_2_indices]["clustering"])

    # creates list with all cells in node_1 and node_2
    attributes_merged_nodes_cell = []
    if type(graph.vs[node_1_indices]["cell"]) == list:
        for cell in graph.vs[node_1_indices]["cell"]:
            attributes_merged_nodes_cell.append(cell)
    else:
        attributes_merged_nodes_cell.append(graph.vs[node_1_indices]["cell"])

    if type(graph.vs[node_2_indices]["cell"]) == list:
        for cell in graph.vs[node_2_indices]["cell"]:
            attributes_merged_nodes_cell.append(cell)
    else:
        attributes_merged_nodes_cell.append(graph.vs[node_2_indices]["cell"])

    # merges node_1 and node_2 by indices, merged attributes are later replace by list of all attributes
    merge_list = [node_1_indices if x == node_2_indices else x for x in vertex_indices]
    graph.contract_vertices(merge_list, combine_attrs="first")

    # deletes the old node no longer connected to the graph
    if node_2_indices < graph.vcount():
        graph.delete_vertices(node_2_indices)

    # assigns the new list to the attributes of the graph
    graph.vs[node_1_indices]["name"] = attributes_merged_nodes_name
    graph.vs[node_1_indices]["clustering"] = attributes_merged_nodes_clustering
    graph.vs[node_1_indices]["cell"] = attributes_merged_nodes_cell

    # remove redundant edges, selects edge-weight based on max.
    graph.simplify(combine_edges=max)

    return graph


def merge_edges_weight_1(graph):
    """
    Merges all edges with edge weight 1.

    :param graph: The graph on which the edges should be merged, iGraph object graph.
    :return: Returns the Graph after changing the edges and nodes after merging the edges.
    """
    edges_to_merge = True
    while edges_to_merge:
        for edge in graph.es:
            if edge["weight"] == 1:
                graph = merge_nodes(graph, edge.source, edge.target)
                edges_to_merge = True
                break
            edges_to_merge = False
    return graph
