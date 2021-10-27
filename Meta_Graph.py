import scanpy as sc
import pandas as pd
import numpy as np
import constclust as cc
import time
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

start = time.time()

# calculate the jaccard index
def jaccard_index(cluster_1, cluster_2):
    set_1 = set(cluster_1)
    set_2 = set(cluster_2)

    intersection = len(set_1.intersection(set_2))
    union = len(set_1) + len(set_2) - intersection

    return np.round(intersection / union, 3)


def read_data(path):
    # read data
    data = pd.read_table(path)

    # print Sample
    print(data[["cell", "C001", "C002", "C003"]].head(3))

    # for testing retrun only a sample of 3 clusters
    return data.iloc[:, 0:3]

    # return data


def max_number_clusters(data):
    max = 0
    # select all columns clusters not the names, to find the highest number of clusters
    # for more generalization may be better with different approach
    for column in data.iloc[:, 1:]:
        max_numer_clusters = data[column].values.max()
        if max < max_numer_clusters:
            max = max_numer_clusters
    max = max + 1
    print("Maximum number of clusters:", max)
    return max


def sort_data_into_cluster_list(data):
    number_clusters = max_number_clusters(data)
    list_clusters_sorted = [[[] for _ in range(number_clusters)] for _ in range(len(data.columns) - 1)]

    for row in data.itertuples():
        cell_name = row[1]
        # index the column number of the data, so index > 1 means cluster one and above are selected
        for index in range(len(row)):
            if index > 1:
                list_clusters_sorted[index - 2][row[index]].append([cell_name, index - 2, row[index]])

    return list_clusters_sorted


def build_graph(cluster_list):
    # create Graph
    G = nx.Graph()

    # add Nodes to graph
    for clustering_used in cluster_list:
        for clusters in clustering_used:
            if clusters:
                name_of_clustering_methode = clusters[0][1]
                name_of_cluster = clusters[0][2]
                G.add_node("C" + str(name_of_clustering_methode) + ":" + str(name_of_cluster))

    # compare every cluster to every cluster except itself, calculate the jaccard index
    # add edges with weight != 0 to graph
    for clustering_A in cluster_list:
        for clustering_B in cluster_list:
            for cluster_A in clustering_A:
                # check if list is empty
                if cluster_A:
                    # get information about cluster
                    clustering_used_A = cluster_A[0][1]
                    name_of_cluster_A = cluster_A[0][2]
                for cluster_B in clustering_B:
                    # check if list is empty
                    if cluster_B:
                        # get information about cluster
                        clustering_used_B = cluster_B[0][1]
                        name_of_cluster_B = cluster_B[0][2]
                        set_A = set([cluster_A[i][0] for i in range(len(cluster_A))])
                        set_B = set([cluster_B[j][0] for j in range(len(cluster_B))])
                        # check if sets are the same
                        node_name_cluster_A = "C" + str(clustering_used_A) + ":" + str(name_of_cluster_A)
                        node_name_cluster_B = "C" + str(clustering_used_B) + ":" + str(name_of_cluster_B)
                        if not node_name_cluster_A == node_name_cluster_B:
                            edge_weight = jaccard_index(set_A, set_B)
                            if edge_weight != 0:
                                G.add_edge(node_name_cluster_A, node_name_cluster_B, weight=edge_weight)

    # drawing the graph with nx
    pos = nx.spring_layout(G)
    labels_edge = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_nodes(G, pos, node_size=150)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_edge)
    nx.draw_networkx_labels(G, pos)
    plt.show()

    # drawing with pyviz
    nt = Network()
    nt.from_nx(G)
    nt.show("network.html")

# sorting the clusters in to lists and building the graph
build_graph(sort_data_into_cluster_list(read_data("s2d1_clustering.tsv")))

# measure the time
end = time.time()
print("Time to run: ", end - start)
