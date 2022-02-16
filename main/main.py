import multires_consensus_clustering as mcc
from pathlib import Path
import time
import scanpy as sc
import pandas as pd
import igraph as ig
from optimzation_using_save_graphs import optimazion_saved_graphs as optimize

HERE = Path(__file__).parent.parent


def run_multires_consensus_clustering(clustering_data, settings_data, adata, plot_edge_weights, plot_labels,
                                      plot_interactive_graph):
    """
    Run the multires consensus clustering by first building a meta_graph for every bin (resolution),
        run community detection on these graphs and merge similiar nodes.
        Create a graph based on all the merged resolution called multi_graph and
        run community detection again to create a merged graph tree representing the clustering resolutions.
    @param plot_interactive_graph: True or False to plot the interactive graph.
    @param plot_labels: True or False to plot the ture labels and the assigned cluster labels.
    @param plot_edge_weights: True or False to plot edge weight distribution.
    @param settings_data: The settings data of the clusterings, number of clusters parameters, etc.
    @param clustering_data: The clustering data based of the adata file, as a pandas dataframe, e.g. cell, C001, ...
    @param adata: The single cell adata file
    @return: The cluster labels create with this function.
    """

    # multi resolution meta graph
    multires_graph = mcc.multiresolution_graph(clustering_data, settings_data, "all", neighbour_based=False)
    print("Multi-graph-build done, Time:", time.time() - start)
    mcc.write_graph_to_file(multires_graph, neighbour_based=False)

    # community detection
    multires_graph = mcc.multires_community_detection(multires_graph, combine_by="first")
    print("Communities detected, Time:", time.time() - start)

    # outlier detection
    mean_edge_weight = mcc.plot_edge_weights(multires_graph, False)
    multires_graph = mcc.hdbscan_outlier(graph=multires_graph, threshold=mean_edge_weight, plot_on_off=False)

    # plot edge weights
    if plot_edge_weights:
        mcc.plot_edge_weights(multires_graph, plot_on_off=True)

    # plot clustering and labels
    if plot_labels:
        true_labels = mcc.true_labels(labels_df=mcc.read_data(HERE / "data\s2d1_labels.tsv", "all"), adata=adata_s2d1)
        cluster_labels =mcc.best_prob_cell_labels(multires_graph, adata=adata_s2d1)

        return cluster_labels

    # plot multi-graph with bokeh
    if plot_interactive_graph:
        mcc.interactive_plot(adata, clustering_data, multires_graph, create_upsetplot=False,
                             create_edge_weight_barchart=False, layout_option="hierarchy")

    # measure the time
    end = time.time()
    print("Time to run: ", end - start)


# run program
if __name__ == "__main__":
    start = time.time()

    # read data
    clustering_data = mcc.read_data(HERE / "data\s2d1_clustering.tsv", "all")
    settings_data = mcc.read_data(HERE / "data\s2d1_settings.tsv", "all")

    # read adata file and select the set used for the clustering.
    adata = sc.read_h5ad(HERE / "data/cite/cite_gex_processed_training.h5ad")
    adata_s2d1 = adata[adata.obs.batch == "s2d1", :].copy()

    print("Read data, Time:", time.time() - start)

    run_multires_consensus_clustering(clustering_data, settings_data, adata=adata_s2d1, plot_edge_weights=False,
                                      plot_labels=False, plot_interactive_graph=False)
