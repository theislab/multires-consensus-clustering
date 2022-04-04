import igraph as ig
import numpy as np
import multires_consensus_clustering as mcc


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


def merge_edges_weight_above_threshold(graph, threshold):
    """
    Merges all edges with edge weight greater or equal to the given threshold.

    @param graph: The graph on which the edges should be merged, iGraph object graph.
    @param threshold: All edges with edge["weight"] >= threshold are merged.
    @return: Returns the Graph after changing the edges and nodes after merging the edges.
    """

    # list for all edges that have an edge weight higher than the threshold
    vertices_to_merge = []

    # add all those edges to the list, using the vertex index of the edge source and target
    for edge in graph.es:
        if edge["weight"] >= threshold:
            source = edge.source
            target = edge.target

            vertices_to_merge.append({source, target})

    # run the stackoverflow function to get all vertices that would be merged into a single on
    sets_to_merge = merge_list_of_sets(vertices_to_merge)

    # convert sets to lists
    list_vertices_to_merge = [list(vertex_group) for vertex_group in sets_to_merge]

    # delete vertices using the vertex clustering function
    graph = merge_vertex_using_clustering(graph, list_vertices_to_merge)

    return graph


def merge_list_of_sets(sets):
    """
    Pull all vertices that would be merged together into a single list. This creates a list of list,
        grouping the vertex into a list of list with the accoridng vertex groups.
    @param sets: List of set in which the edges are contained [{0,1},{1,4}{2,3},...]
    @return: Returns a list of list contains all vertices that would be merged into a single on.
        E.g [[0,1,4], [2,3,...], ...]
    Function from:
    https://stackoverflow.com/questions/9110837/python-simple-list-merging-based-on-intersections
    """
    merged = True
    while merged:
        merged = False
        results = []
        while sets:
            common, rest = sets[0], sets[1:]
            sets = []
            for x in rest:
                if x.isdisjoint(common):
                    sets.append(x)
                else:
                    merged = True
                    common |= x
            results.append(common)
        sets = results
    return sets


def merge_vertex_using_clustering(graph, merger_list):
    """
    Create a vertex clustering from the list for each vertex group with in and merge those vertices.
    @param graph: iGraph graph object on which the vertices should be merged.
    @param merger_list: The list of the vertex groups that should be merged into a single node.
        E.g. [[0,1,3,11...,], [7,8,18,...], ... ]
    @return: The merged graph.
    """

    # assign each vertex a unique number before the merger
    graph.vs["merge_edges"] = range(- graph.vcount(), 0)

    # set vertex group index
    index_merger = 1

    # iterate trough each group
    for vertex_group in merger_list:

        # iterate through each vertex in the group
        for vertex_index in vertex_group:
            # assign all vertices in the group the same merge_edges index to create a clustering
            graph.vs[vertex_index]["merge_edges"] = index_merger

        index_merger += 1

    # convert the merge_edges index into clustering to be merged together
    vertex_clustering = ig.clustering.VertexClustering.FromAttribute(graph, attribute="merge_edges")

    # use the merge by list function to cluster the graph and reassign the attributes correctly
    merged_graph = mcc.merge_by_list(vertex_clustering)

    # delete vertex attribute used for merging the edges
    del (merged_graph.vs["merge_edges"])

    return merged_graph
