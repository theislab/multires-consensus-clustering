import igraph as ig
import networkx
import numpy as np
from multires_consensus_clustering import Meta_Graph as mg
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from igraph.clustering import VertexClustering


def merg_edges_weight_1(G):
    """
    Merges all edges with edge weight 1.

    :param G: The graph on which the edges should be merged, iGraph object graph.
    :return: Returns the Graph after changing the edges and nodes after merging the edges.
    """
    for edge in G.es:
        if edge["weight"] == 1:
            vertex_indices = range(G.vcount())
            merge_list = [edge.source if x == edge.target else x for x in vertex_indices]
            G.contract_vertices(merge_list, combine_attrs="first")

    vertices_to_delete = [vertex.index for vertex in G.vs if vertex['name'] is None]
    G.delete_vertices(vertices_to_delete)
    G.simplify(combine_edges=max)

    return G

def min_cuts(G):
    cut_value_0 = 0
    weight_normalized = np.sum(G.es["weight"]) / G.ecount()
    print(weight_normalized)
    size_partition_1 = 0
    size_partition_2 = 1
    print(G.ecount())
    while size_partition_1 < size_partition_2:
        cut = G.mincut(source=None, target=None, capacity="weight")
        G.delete_vertices(cut.partition[0])
        size_partition_1 = len(cut.partition[0])
        size_partition_2 = len(cut.partition[1])
        print(cut.value)
    print(G.ecount())
    mg.plot_graph(G, "label_on", "degree")


def graph_community_detection(G):
    graph = ig.Graph.community_infomap(G, edge_weights="weight")
    ig.plot(graph)
    graph = ig.Graph.community_label_propagation(G, weights="weight")
    ig.plot(graph)
    graph = ig.Graph.community_leading_eigenvector(G, weights="weight")
    ig.plot(graph)
    graph = ig.Graph.community_multilevel(G, weights="weight")
    ig.plot(graph)
    graph = ig.Graph.community_spinglass(G, weights="weight")
    ig.plot(graph)