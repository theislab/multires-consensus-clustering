import pyvis.network
from numba import njit, types
import h5py
import time
import igraph as ig
import numpy as np
import pandas as pd
import scanpy as sc
import itertools
import seaborn as sns
from bokeh.plotting import figure
from bokeh.models import GraphRenderer
from bokeh.palettes import Spectral8

start = time.time()


# relabeling clusters by Isaac
def relabel_clusterings(clusterings: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure each cluster in a set of clusterings has a unique integer ID.

    Each clustering should have a contiguous range of integers for it's cluster's IDs.
    """
    clusterings = clusterings.rank(axis=0, method="dense").astype(int) - 1
    cvals = clusterings.values
    cvals[:, 1:] += (cvals[:, :-1].max(axis=0) + 1).cumsum()
    return clusterings


# jaccrad index computation by Isaac
@njit
def clustering_edges_array(clustering1: "np.ndarray[int]",
                           clustering2: "np.ndarray[int]") -> "list[tuple[int, int, float]]":
    """
    Find jaccard similarities between clusters in two clusterings.

    Params
    ------
    c1, c2
        Clusterings encoded as array of integers. Assumes each cluster has a unique integer id (will be node label in graph).

    Returns
    -------

    Edges in the cluster-cluster graph encoded as tuples of (node1, node2, jaccard similarity).
    """
    edges = []
    offset1 = clustering1.min()
    offset2 = clustering2.min()
    # Because of how I've done unique node names, potentially this
    # could be done in a more generic way by creating a mapping here.
    offset_clusts1 = clustering1 - offset1
    offset_clusts2 = clustering2 - offset2
    # Allocate coincidence matrix
    nclusts1 = offset_clusts1.max() + 1
    nclusts2 = offset_clusts2.max() + 1
    coincidence = np.zeros((nclusts1, nclusts2))
    # Allocate cluster size arrays
    ncells1 = np.zeros(nclusts1)
    ncells2 = np.zeros(nclusts2)
    # Compute lengths of the intersects
    for cell in range(len(clustering1)):
        c1 = offset_clusts1[cell]
        c2 = offset_clusts2[cell]
        # Coincidence matrix stores the size of the intersection of the clusters
        coincidence[c1, c2] += 1
        ncells1[c1] += 1
        ncells2[c2] += 1
    for cidx1, cidx2 in np.ndindex(coincidence.shape):
        isize = coincidence[cidx1, cidx2]
        if isize < 1:
            continue
        # Jaccard similarity is |intersection| / |union|.
        # Here, |union| is calculated as |clustering1| + |clustering2|
        jaccard_sim = isize / (ncells1[cidx1] + ncells2[cidx2] - isize)
        edge = (cidx1 + offset1, cidx2 + offset2, jaccard_sim)
        edges.append(edge)
    return edges


# read data
def read_data(path, sample_size):
    data = pd.read_table(path)

    # for testing return only a sample of clusters
    if type(sample_size) == int:
        return data.iloc[:, 0:sample_size]

    # return complete set
    if sample_size == "all":
        return data


def build_graph(data, color_vertex_methode, label_on_off):
    # create graph
    G = ig.Graph()

    # exclude cell names from data so later all possible combination are the combinations of clusters
    clusters = data.loc[:, data.columns != 'cell']

    # all possible combinations of clusters
    combinations_of_clusters = list(itertools.combinations(clusters.columns, 2))
    print("Number of possible cluster combinations ", len(combinations_of_clusters))

    vertex_labels = []
    vertex_cluster_methode = []
    edge_labels = []
    for cluster_methode in combinations_of_clusters:

        # name of the clustering methode
        cluster_methode_0 = cluster_methode[0]
        cluster_methode_1 = cluster_methode[1]

        # name of the cluster
        cluster_0 = clusters[cluster_methode_0]
        cluster_1 = clusters[cluster_methode_1]

        # compute the jaccard index using Isaacs code, returns a list with start, end and weight of an edge
        edge_list = clustering_edges_array(cluster_0.values, cluster_1.values)

        for edge in edge_list:
            # name of the edges and edge weight
            edge_start = cluster_methode_0 + ":" + str(edge[0])
            edge_end = cluster_methode_1 + ":" + str(edge[1])
            edge_weight = edge[2]

            # check if node for cluster_0 is already in the graph
            if not (edge_start in vertex_labels):
                G.add_vertices(edge_start)
                vertex_labels.append(edge_start)
                vertex_cluster_methode.append(cluster_methode_0)

            # check if node for cluster_1 is already in the graph
            if not (edge_end in vertex_labels):
                G.add_vertices(edge_end)
                vertex_labels.append(edge_end)
                vertex_cluster_methode.append(cluster_methode_1)

            # check if edge weight is zero
            if edge_weight != 0:
                # add edge to graph
                G.add_edges([(edge_start, edge_end)])
                edge_labels.append(np.round(edge_weight, 5))

        # add edge weight and name to graph
        G.es["weight"] = edge_labels
        G.vs["name"] = vertex_labels

    # color by degree
    if color_vertex_methode == "degree":
        # calculate the degree of each vertex
        degree_graph = []
        for vertex in vertex_labels:
            degree_graph.append(G.degree(vertex))

        # color according to the degree
        color_palette = sns.color_palette("viridis", n_colors=max(degree_graph) + 1)
        colors = [color_palette[degree] for degree in degree_graph]
        G.vs['color'] = colors

    # color by clustering
    if color_vertex_methode == "clustering":

        # assign number to each clustering methode, e.g. "C001" = 0
        names_to_numbers = {}
        index_names = 0
        for name in set(vertex_cluster_methode):
            names_to_numbers[name] = index_names
            index_names = index_names + 1

        color_palette = ig.ClusterColoringPalette(len(clusters.columns))
        # color the nodes by clustering methode using the numbers, e.g. "C001" = 0
        colors = [color_palette[names_to_numbers[name]] for name in vertex_cluster_methode]
        G.vs['color'] = colors

    # layout = G.layout_fruchterman_reingold()
    # layout = G.layout_kamada_kawai()
    layout = G.layout_auto()

    # turn labels on or off
    if label_on_off == "label_on":
        ig.plot(G, layout=layout, vertex_size=20, vertex_label=vertex_labels, edge_label=edge_labels)
    if label_on_off == "label_off":
        ig.plot(G, layout=layout, vertex_size=20)

    return G


# returns a list of all clusterings closest two the given number of clusters or with the number of cluster (if >= 2)
def sort_by_number_clusters(settings, data, number_of_clusters):
    # transforms settings to a dataframe with occurrence of the number of clusters
    frequency = settings.groupby('n_clusters').count()
    reindex_frequency = frequency.reset_index()

    # check if number_of_clusters exists in the list of clustering settings
    if not (number_of_clusters in np.unique(settings['n_clusters'].values)):
        # if not take the two clusters two clusters closest to the number
        closest_clusterings = reindex_frequency.iloc[
            (reindex_frequency['n_clusters'] - number_of_clusters).abs().argsort()[:2]]

        # number of cluster for the closest clusterings
        number_of_clusters_closest = closest_clusterings.loc[:, "n_clusters"].values
        # find all clusterings with that number of clusters
        list_of_clusterings = settings.loc[settings['n_clusters'].isin(number_of_clusters_closest)]["id"].values

        print("There are no clustering methods with ", number_of_clusters,
              " clusters; the two closest clusterings are :", list_of_clusterings)

    # number of clusters exist in settings
    else:
        # check if number of clusters occurs only once
        if frequency.loc[number_of_clusters, "id"] == 1:
            # if so take the two closest clusters of the selected number of clusters
            closest_clusterings = reindex_frequency.iloc[
                (reindex_frequency['n_clusters'] - number_of_clusters).abs().argsort()[:2]]

            # number of cluster for the closest clusterings
            number_of_clusters_closest = closest_clusterings.loc[:, "n_clusters"].values
            # find all clusterings with that number of clusters
            list_of_clusterings = settings.loc[settings['n_clusters'].isin(number_of_clusters_closest)]["id"].values

            print("Clustering methods with ", number_of_clusters,
                  " clusters  occurs only once; the two closest clusterings are :", list_of_clusterings)

        # otherwise there are two or more clusterings with the same number of clusters

        # number of clusters exists more than once
        else:
            # select all clusterings with the given number of clusters
            list_of_clusterings = settings.loc[settings['n_clusters'] == number_of_clusters]["id"].values

            print("Clustering methods with ", number_of_clusters, " clusters:", list_of_clusterings)

    # return the list with all clusterings
    return data[list_of_clusterings]


def merge_vertices(G, edge):
    mapping = {}
    source_vertex = graph.vs[edge.source]
    target_vertex = graph.vs[edge.target]
    mapping[source_vertex] = source_vertex
    mapping[target_vertex] = source_vertex
    # G.contract_vertices(mapping)


def graph_analysis(G, analysis_methode):
    if analysis_methode == "clique_number":
        print(G.clique_number())

    if analysis_methode == "girth":
        print(G.girth())

    if analysis_methode == "decompose":
        decompose_G = G.decompose()
        ig.plot(decompose_G[0])

    if analysis_methode == "maximal_cliques":
        print(G.cliques())

    if analysis_methode == "merge_weight_1":
        for edge in G.es:
            if edge["weight"] == 1:
                merge_vertices(G, edge)


# run program
# read data
clustering_data = read_data("s2d1_clustering.tsv", "all")
settings_data = read_data("s2d1_settings.tsv", "all")

# build graph, G is used as the variable for the Graph internally
number_of_clusters_data = sort_by_number_clusters(settings_data, clustering_data, 3)
graph = build_graph(number_of_clusters_data, "degree", "off")

# analyse graph
# graph_analysis(graph, "maximal_cliques")
merge_edges(graph)

# measure the time
end = time.time()
print("Time to run: ", end - start)
