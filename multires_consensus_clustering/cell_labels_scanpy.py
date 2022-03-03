import pandas
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent.parent


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


def graph_to_cell_labels_df(graph):
    """
    Iterates through all nodes to find the best probability tho assign a cell label,
        if the probabilities are equal chooses the highest resolution.
    @param graph: The graph from which the probabilities are generated, iGraph.
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

    return df_cell_clusters


def df_cell_clusters_to_labels(label_df, adata, plot_labels):
    """
    Takes the given data frame of the cells, labels and probabilities and return only the labels.
        Can also plot those labels.
    @param label_df: Pandas dataframe with cell names as index, clustering labels and probabilities.
    @param plot_labels: True or False. Turns the label plot on or off.
    @return: Returns the cluster labels.
    """

    cluster_labels = label_df["cluster_labels"]

    # plot labels
    if plot_labels:
        adata.obs["mcc_cluster_labels"] = cluster_labels
        number_clusters = np.unique(cluster_labels.values)
        print("The graph has ", len(number_clusters) + 1, "clusters")

        plot = sc.pl.umap(adata, color=["mcc_cluster_labels"], show=True)

    return cluster_labels
