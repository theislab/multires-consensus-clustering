from numba import njit, types
import h5py
import time
import igraph as ig
import numpy as np
import pandas as pd
import itertools
import seaborn as sns

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
    """
    Reads the data from file and converts it to a pandas df, returns a sample or the complete data.

    :param path: Path of the data.
    :param sample_size: If not all data selected returns only a smaller sample for testing, type int or string ,
        number of samples (int) or "all".
    :return: Data as a pandas dataframe
    """
    data = pd.read_table(path)

    # for testing return only a sample of clusters
    if type(sample_size) == int:
        return data.iloc[:, 0:sample_size]

    # return complete set
    if sample_size == "all":
        return data


def build_graph(data):
    """
    Builds a meta-graph based on the given data (list of clusterings).

    Assigns each cluster for each clustering a node in the graph and calculates the edges (weighted) using the jaccard index.
    :param data: Pandas dataframe from the single cell data.
    :return: Returns an igraph object Graph.
    """
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
    G.vs["clustering"] = vertex_cluster_methode

    return G


# returns a list of all clusterings closest two the given number of clusters or with the number of cluster (if >= 2)
def sort_by_number_clusters(settings, data, number_of_clusters):
    """
    Binning of the clusterings by number of clusters.

    :param settings: Setting data from the sc clustering as a pandas df.
    :param data: Clutering data from the sc data as a pandas df.
    :param number_of_clusters: The number of clusters which should be binned together.
    :return: Returns a list of the clusterings binned by the number of clusters, e.g. [C001, C002, ...].
    """

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



def graph_analysis(G, analysis_methode):
    """
    A function for analysing the graph using different methods. At the moment more of a search for useful analysis.
    E.g. merges all edges with edge weight 1, if analysis_methode = "merge_weight_1".

    :param G: The graph which to analyse, iGraph object graph.
    :param analysis_methode: The name of the function for the analysis, type string, e.g. "merge_weight_1".
    :return: Returns the Graph after changing the edges or nodes depending on the chosen function.
    """
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
                vertex_indices = range(G.vcount())
                merge_list = [edge.source if x == edge.target else x for x in vertex_indices]
                G.contract_vertices(merge_list, combine_attrs="first")

        vertices_to_delete = [vertex.index for vertex in G.vs if vertex['name'] is None]
        G.delete_vertices(vertices_to_delete)
        G.simplify(combine_edges=max)

        return G


def plot_graph(G, label_on_off, color_vertex):
    """
    Function for plotting the Graph for all other used functions.

    :param G: Graph that should be plotted.
    :param label_on_off: Turns label on or off, get "label_off" or "label_on", type: string.
    :param color_vertex: Gets the coloring method for the vertices, "degree" or "clustering", type: string.
    :return: No return; only plots the graph.
    """

    # color by degree
    if color_vertex == "degree":
        # calculate the degree of each vertex
        degree_graph = []
        for vertex in G.vs["name"]:
            degree_graph.append(G.degree(vertex))

        # color according to the degree
        color_palette = sns.color_palette("viridis", n_colors=max(degree_graph) + 1)
        colors = [color_palette[degree] for degree in degree_graph]
        G.vs['color'] = colors

    # color by clustering
    if color_vertex == "clustering":
        # assign number to each clustering methode, e.g. "C001" = 0
        names_to_numbers = {}
        index_names = 0
        for name in set(G.vs["clustering"]):
            names_to_numbers[name] = index_names
            index_names = index_names + 1
        color_palette = ig.ClusterColoringPalette(len(names_to_numbers))
        # color the nodes by clustering methode using the numbers, e.g. "C001" = 0
        colors = [color_palette[names_to_numbers[name[0:4]]] for name in G.vs["name"]]
        G.vs['color'] = colors

    # layout = G.layout_fruchterman_reingold()
    # layout = G.layout_kamada_kawai()
    layout = G.layout_auto()

    # turn labels on or off
    if label_on_off == "label_on":
        ig.plot(G, layout=layout, vertex_size=20, vertex_label=G.vs["name"], edge_label=G.es["weight"])
    if label_on_off == "label_off":
        ig.plot(G, layout=layout, vertex_size=20)


# run program
# read data
clustering_data = read_data("s2d1_clustering.tsv", "all")
settings_data = read_data("s2d1_settings.tsv", "all")

# build graph, G is used as the variable for the Graph internally
number_of_clusters_data = sort_by_number_clusters(settings_data, clustering_data, 3)
graph = build_graph(number_of_clusters_data)

# analyse graph
G = graph_analysis(graph, "merge_weight_1")
plot_graph(G, "label_on", "degree")
# merge_edges(graph)

# measure the time
end = time.time()
print("Time to run: ", end - start)

