import multires_consensus_clustering as mcc
from pathlib import Path
import time
import scanpy as sc
import pandas as pd
import igraph as ig
import numpy as np

HERE = Path(__file__).parent.parent


def run_multires_consensus_clustering(clustering_data, settings_data, adata, plot_labels,
                                      plot_interactive_graph, community_mulit_res, connect_graph_neighbour_based,
                                      multi_resolution):
    """
    Run the multi-res consensus clustering by first building a meta_graph for every bin (resolution),
        run community detection on these graphs and merge similar nodes.
        Create a graph based on all the merged resolution called multi_graph and
        run community detection again to create a merged graph tree representing the clustering resolutions.

    @param multi_resolution:
    @param settings_data: The settings data of the clusterings, number of clusters parameters, etc.
    @param clustering_data: The clustering data based of the adata file, as a pandas dataframe, e.g. cell, C001, ...
    @param adata: The single cell adata file

    @param connect_graph_neighbour_based: True or False, connects the multi-resolution-graph either
        by neighbouring resolutions or connects all vertices. HDBscan does not work with the neighbour based graph!
    @param community_mulit_res: "leiden", "hdbscan", "component" or otherwise automatically louvain.

    @param plot_interactive_graph:
    @param plot_labels: True or False to plot the ture labels and the assigned cluster labels.

    @return: The cluster labels create with this function.
    """

    # multi resolution meta graph
    multires_graph = mcc.multiresolution_graph(clustering_data, settings_data, "all",
                                               neighbour_based=connect_graph_neighbour_based)
    print("Multi-graph-build done, Time:", time.time() - start)

    # community detection
    if connect_graph_neighbour_based == True and community_mulit_res == "hdbscan":
        print("Neighbor based graph does not support interaction with HDBscan.")
        multires_graph = mcc.multires_community_detection(multires_graph, community_detection="leiden",
                                                          clustering_data=clustering_data, multi_resolution=multi_resolution)
    else:
        multires_graph = mcc.multires_community_detection(multires_graph, community_detection=community_mulit_res,
                                                          clustering_data=clustering_data, multi_resolution=multi_resolution)
    print("Communities detected, Time:", time.time() - start)

    # create clustering labels and plot them if plot_labels == True
    df_clusters = mcc.graph_to_cell_labels_df(multires_graph)
    cluster_labels = mcc.df_cell_clusters_to_labels(df_clusters, adata, plot_labels)

    # plot multi-graph with bokeh
    if plot_interactive_graph:
        mcc.interactive_plot(adata, clustering_data, multires_graph, create_upsetplot=False,
                             create_edge_weight_barchart=False, layout_option="hierarchy")

    # calculates a final probability score for all cells
    df_clusters = mcc.uncertainty_measure_cells(multires_graph)

    mcc.probability_umap_plot(df_clusters, adata)

    # measure the time
    end = time.time()
    print("Time to run: ", end - start)

    return cluster_labels


# run program
if __name__ == "__main__":
    start = time.time()

    # read data
    clustering_data = mcc.read_data(HERE / "data\s2d1_clustering.tsv")
    settings_data = mcc.read_data(HERE / "data\s2d1_settings.tsv")
    labels_data = mcc.read_data(HERE / "data\s2d1_labels.tsv")

    # read adata file and select the set used for the clustering.
    adata = sc.read_h5ad(HERE / "data/cite/cite_gex_processed_training.h5ad")
    adata = adata[adata.obs.batch == "s2d1", :].copy()

    # check if adata file has calculated the umap plot
    if "X_umap" not in adata.obsm_keys():
        sc.tl.umap(adata)

    print("Read data, Time:", time.time() - start)

    run_multires_consensus_clustering(clustering_data, settings_data, adata=adata, community_mulit_res="leiden",
                                      connect_graph_neighbour_based=False, plot_labels=True, plot_interactive_graph=False,
                                      multi_resolution=1)

    """
    # run evaluation for the multi-res parameter
    mcc.cluster_consistency(adata, clustering_data, settings_data, labels_data)
    """