import igraph as ig
import numpy as np
from multires_consensus_clustering import Meta_Graph as mg





def min_cuts(G):
    """
    Function for the min-cut algorithm. Separates the graph in two community by separating the edges with
     lowest weight. Repeats the process until condition for max separations is met.


    @param G: The Graph on which the min-cut algorithmen is proformed
    @return: Returns the cut graph.
    """
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

    return G

def graph_community_detection(G):
    """
    Function for community detection. Uses the igraph community detection functions to partition the graph
     in community based on edge weight. Merges the community to one node, by combing all attributes in a list and
     by mean edge weight.

    @param G: The graph on which to detect communities.
    @return: Returns the merged graph based on community.
    """
    graph = ig.Graph.community_infomap(G, edge_weights="weight").cluster_graph(combine_vertices=list, combine_edges=np.mean)

    """
    ig.plot(graph, vertex_color=ig.drawing.colors.ClusterColoringPalette(graph.vcount()),
            edge_label=np.round(graph.es["weight"],5))
    
    graph = ig.Graph.community_label_propagation(G, weights="weight")
    ig.plot(graph)
    graph = ig.Graph.community_leading_eigenvector(G, weights="weight")
    ig.plot(graph)
    graph = ig.Graph.community_multilevel(G, weights="weight")
    ig.plot(graph)
    graph = ig.Graph.community_spinglass(G, weights="weight")
    ig.plot(graph)
    """

    return graph