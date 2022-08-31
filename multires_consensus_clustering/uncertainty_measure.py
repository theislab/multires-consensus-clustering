import igraph as ig
import numpy as np
import scanpy as sc
import pandas as pd
import multires_consensus_clustering as mcc
import matplotlib.pyplot as plt


def uncertainty_measure_cells(graph, adata, plot):
    """
    Calculates a probability score for all cells based on the probabilities in each node
        and the number the cell occurs with the same probability throughout the graph.

    @param graph: The graph on with the clustering is based.
    @param adata: The adata file from which the clustering is generated.
    @param plot: Turns the plot on or off with True or False.

    @return: A pandas dataframe with the probabilities.
    """

    # get df of the cluster labels (contains all infos about the meta-graph)
    # cell_occurrence, cell_occurrence with the same probabilities, original probabilities
    df_clusters = mcc.graph_to_cell_labels_df(graph)

    # get the values for the cells
    cell_occurrence_node = df_clusters["number_cell_occurrence"].values
    cell_occurrence_prob = df_clusters["number_same_probability"].values
    cell_probabilities = df_clusters["probability"].values

    # divide the number a cell occurs with the same probability by the time it occurs in total
    zip_occurrence = zip(cell_occurrence_prob, cell_occurrence_node)
    occurrence = [occur[0] / occur[1] if occur[1] != 0 else 1 for occur in zip_occurrence]

    # multiple the probabilities if the occurrence is not zero
    zip_probs = zip(cell_probabilities, occurrence)
    estimator = [prob[0] * prob[1] if prob[1] != 0 else prob[0] for prob in zip_probs]

    # add new probabilities to the dataframe
    df_clusters["probability"] = estimator

    # plot the probabilities with umap
    if plot:
        adata.obs["mcc_cluster_probabilities"] = estimator
        plt.tight_layout()
        sc.pl.umap(adata, color=["mcc_cluster_probabilities"], show=True)

    return df_clusters

