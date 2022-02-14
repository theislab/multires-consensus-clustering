import pandas
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path


HERE = Path(__file__).parent.parent


def graph_nodes_cells_to_df(graph, clustering_data):
    """
    Create as pandas dataframe with the probability of each cell occurring in one node.

    @param graph: The graph, need attribute graph.vs["cell"]
    @param clustering_data: The clustering data, type pandas.dataframe, e.g. cell, C001, C002, ...
    @return: Return a pandas dataframe with cells as index and probability sorted by node.
    """
    label_list = []

    for merged_vertices in graph.vs["cell"]:
        label_list.append(sum(merged_vertices, []))

    # create df for the probability values
    graph_df = pd.DataFrame()
    graph_df["cell"] = clustering_data["cell"]
    graph_df.set_index("cell", inplace=True)

    vertex_index = 0
    for merged_node in label_list:
        # count cells
        cell_names_vertex, counts = np.unique(merged_node, return_counts=True)
        # calculate the probability
        counts = counts / len(graph.vs[vertex_index]["name"])

        # creates new column in df
        df_probabilities_vertex = pd.DataFrame({'cell': cell_names_vertex,
                            'merged_node_' + str(vertex_index): counts})
        df_probabilities_vertex.set_index("cell", inplace=True)

        # assigns the values to the cells
        graph_df = pd.concat([graph_df, df_probabilities_vertex], axis=1)

        vertex_index +=1

    # change NaN values to 0
    graph_df = graph_df.fillna(0)

    return graph_df


def single_node_to_df(vertex, clustering_data):
    """
    Create as pandas dataframe with the probability of each cell occurring in the given node.

    @param vertex: iGraph vertex (node), needs attribute ["cell"]
    @param clustering_data: The clustering data, type pandas.dataframe, e.g. cell, C001, C002, ...
    @return: Return a pandas dataframe with cells as index and probability sorted by node.
    """
    # create list with all cell names in vertex
    cluster_cell_list = sum(vertex["cell"], [])

    # create df for the probability values
    graph_df = pd.DataFrame()
    graph_df["cell"] = clustering_data["cell"]
    graph_df.set_index("cell", inplace=True)

    # count cells
    cell_names_vertex, counts = np.unique(cluster_cell_list, return_counts=True)
    # calculate the probability
    counts = counts / len(vertex["name"])

    # creates new column in df
    df_probabilities_vertex = pd.DataFrame({'cell': cell_names_vertex,
                                            'merged_node': counts})
    df_probabilities_vertex.set_index("cell", inplace=True)

    # assigns the values to the cells
    graph_df = pd.concat([graph_df, df_probabilities_vertex], axis=1)

    # change NaN values to 0
    graph_df = graph_df.fillna(0)

    return graph_df


def relabel_cell(df_cell_probability, adata_s2d1):
    """
    Assigns all cell labels from the clustering the original cell_type from the adata (TCATTCAGTCACCGAC-1-s2d1 -> Erythroblast).
    If cell labels from clustering is not in adata, assigns "unknown".

    @param df_cell_probability: Pandas dataframe with cells as rows, merged nodes as clusters and
        values the probability of cell being in the node (0 to 1).
    @param adata_s2d1: The adata set on which the clustering is based.
    @return: Pandas dataframe with cell_types as rows and a summed up probability as values (weighted count by probability).
    """

    cell_type_dic = {}
    cell_index_counter = 0
    cell_index = adata_s2d1.obs["cell_type"].index

    labels_not_in_adata = set(df_cell_probability.index) - set(adata_s2d1.obs["cell_type"].index)

    for cell_name in adata_s2d1.obs["cell_type"]:
        cell_type_dic[cell_index[cell_index_counter]] = cell_name
        cell_index_counter += 1

    for cell_name in labels_not_in_adata:
        cell_type_dic[cell_name] = "unknown"

    df_cell_probability = df_cell_probability.rename(cell_type_dic)

    df_cell_probability = df_cell_probability.groupby(['cell']).sum()

    return df_cell_probability


def assign_cluster_to_cell(df_cell_probability):
    """
    Returns clustering labels based on the highest probability.
    Creates a pandas dataframe with cells names as index and clustering labels in the column.

    @return df_cell_clusters: Pandas dataframe with cell names as index and clustering labels as column.
    """

    df_cell_clusters = pd.DataFrame({
        'cell': df_cell_probability.index,
        'probability': pd.Series([-1] * len(df_cell_probability), dtype='float'),
        'cluster_labels': [str(0)] * len(df_cell_probability)
    })

    df_cell_clusters['probability'].astype(dtype='float64')

    df_cell_clusters.set_index('cell', inplace=True, drop=True)

    cluster_label = 1
    for column in df_cell_probability.columns:
        index_row = 0
        for cell_probability in df_cell_probability[column]:
            if df_cell_clusters["probability"][index_row] < cell_probability and 0 < cell_probability:
                df_cell_clusters.iat[index_row, 0] = cell_probability
                df_cell_clusters.iat[index_row, 1] = str(cluster_label)
            index_row += 1
        cluster_label += 1

    print("Certainty cluster labels:", df_cell_clusters["probability"].sum() / len(df_cell_clusters))

    return df_cell_clusters


def graph_to_clustering(graph, adata, cluster_or_probability):
    """
    Creates cluster labels for the adata set using the df in each node of the graph and
        creates a static UMAP plot colored by cluster.

    @param cluster_or_probability: Plot either the cluster labels or the probabilities; "cluster" or "probability"
    @param adata: The cell data on which the graph is based.
    @param graph: The graph on which the clusters are based, iGraph graph. Vertices need the attribute "probability_df",
        graph.vs["probability_df"].
    @return: Returns the clustering labels as a pandas dataframe.
    """

    # check if "probability_df" is an attribute of the graph vertices
    if "probability_df" in graph.vs.attributes():
        selected_resolution_nodes = select_resolution_nodes(graph, resolution_level=-1)
        graph_clustering_df = pandas.DataFrame()
        graph_clustering_df["cell"] = graph.vs[0]["cell_index"]
        i = 0

        # create a dataframe with the probabilities of all nodes
        for vertex in selected_resolution_nodes:
            graph_clustering_df["vertex_end_of_tree" + "_" + str(i)] = vertex["probability_df"]
            i += 1
        graph_clustering_df.set_index("cell", inplace=True)

        # generate a single dataframe with probabilities and cluster labels
        clustering_labels = assign_cluster_to_cell(graph_clustering_df)

        # plot data
        # plot clusters
        if cluster_or_probability == "cluster":
            adata.obs["meta_clusters"] = clustering_labels["cluster_labels"]

            number_clusters = np.unique(clustering_labels["cluster_labels"].values)
            print("The graph has ", len(number_clusters) + 1, "clusters")

            plot = sc.pl.umap(adata, color=["meta_clusters"], show=True)

        # plot probabilities
        elif cluster_or_probability == "probability":
            adata.obs["meta_clusters"] = clustering_labels["probability"]

            plot = sc.pl.umap(adata, color=["meta_clusters"], show=True)

        return clustering_labels


def select_resolution_nodes(graph, resolution_level):
    """
    Selects nodes based on the lowest branch, second lowest, etc. And returns all such nodes in  a list.

    @param resolution_level: Which level should be selected. The lowest level is 0, the second lowest -1, etc.
    @param graph: The graph, has to be a tree structure where a node has only one edge from a higher node, iGraph graph.
    @return: list_selected_nodes, a list of all the nodes within the lowest, etc. part of the branch.
    """
    list_selected_nodes = []

    for vertex in graph.vs:
        all_edges_vertex = vertex.all_edges()
        number_edge_vertex = len(all_edges_vertex)

        if resolution_level == 0:
            if number_edge_vertex == 1:
                # the end of a branch (of the tree) has only one edge in/out
                # -> the target of the edge is the lowest node of the branch
                list_selected_nodes.append(graph.vs[all_edges_vertex[0].target])

        elif resolution_level == -1:
            # chooses the source of the edge instead of the target to get the second lowest
            if number_edge_vertex == 1:
                list_selected_nodes.append(graph.vs[all_edges_vertex[0].source])

    return list_selected_nodes


def true_labels(labels_df, adata):
    """
    Assigns the 'true' labels to the adata plot based on the labels_df provided.
    @param labels_df: Dataframe containg the "true" labels, for benachmark data with provided labels.
    @param adata: The single cell data, as an adata file.
    """

    label_dict = dict(zip(labels_df["cell"], labels_df["cell_type"]))
    new_labels = [label_dict[cell] for cell in list(adata.obs.index)]

    adata.obs["true_labels"] = new_labels

    print("The graph has ", len(np.unique(labels_df["cell_type"])) + 1, "cell types")

    plot = sc.pl.umap(adata, color=["true_labels"], show=True)


def best_prob_cell_labels(graph, adata):
    """
    Iterates through all nodes to find the best probability tho assign a cell label,
        if the probabilities are equal chooses the highest resolution.
    @param graph: The graph from which the probabilies are generated, iGraph.
    @return: The clustering labels.
    """
    len_df = len(graph.vs[0]["probability_df"])
    df_cell_clusters = pd.DataFrame({
        'cell': graph.vs[0]["cell_index"],
        'probability': pd.Series([-1] * len_df, dtype='float'),
        'cluster_labels': [str(0)] * len_df,
        'level_cluster_label': [np.inf] * len_df,
    })

    cluster_label_index = 0
    for vertex in graph.vs:
        vertex_probabilities = vertex["probability_df"]
        for index_df in range(len_df):
            probability_cell_vertex = vertex_probabilities[index_df]
            probability_cell_label_df = df_cell_clusters.iat[index_df, 1]
            vertex_level = vertex["level"]

            # choose the best probability for the cluster out of all vertices
            if probability_cell_label_df < probability_cell_vertex:
                df_cell_clusters.iat[index_df, 1] = probability_cell_vertex
                df_cell_clusters.iat[index_df, 2] = str(cluster_label_index)
                df_cell_clusters.iat[index_df, 3] = vertex_level

            # if they are the same choose the highest resolution
            elif probability_cell_label_df == probability_cell_vertex:
                df_cell_clusters.iat[index_df, 2] = str(cluster_label_index)
                df_cell_clusters.iat[index_df, 3] = vertex_level

        cluster_label_index += 1

    # replace index with cell names
    df_cell_clusters.set_index("cell", inplace=True)

    # plot labels
    adata.obs["meta_clusters"] = df_cell_clusters["cluster_labels"]
    number_clusters = np.unique(df_cell_clusters["cluster_labels"].values)
    print("The graph has ", len(number_clusters) + 1, "clusters")

    plot = sc.pl.umap(adata, color=["meta_clusters"], show=True)

    return df_cell_clusters["cluster_labels"]