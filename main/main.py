import multires_consensus_clustering as mcc
from pathlib import Path
import time
import scanpy as sc
import pandas as pd
import igraph as ig
import numpy as np
from optimization_using_saved_graphs import optimization_saved_graphs as opti

HERE = Path(__file__).parent.parent


def run_multires_consensus_clustering(clustering_data, settings_data, adata, plot_labels,
                                      plot_interactive_graph, community_mulit_res, outlier_threshold,
                                      merge_edges_threshold, outlier_mulit_res, connect_graph_neighbour_based):
    """
    Run the multi-res consensus clustering by first building a meta_graph for every bin (resolution),
        run community detection on these graphs and merge similar nodes.
        Create a graph based on all the merged resolution called multi_graph and
        run community detection again to create a merged graph tree representing the clustering resolutions.

    @param settings_data: The settings data of the clusterings, number of clusters parameters, etc.
    @param clustering_data: The clustering data based of the adata file, as a pandas dataframe, e.g. cell, C001, ...
    @param adata: The single cell adata file

    @param connect_graph_neighbour_based: True or False, connects the multi-resolution-graph either
        by neighbouring resolutions or connects all vertices. HDBscan does not work with the neighbour based graph!
    @param community_mulit_res: "leiden", "hdbscan", "component" or otherwise automatically louvain.

    @param outlier_mulit_res: "probability" or "hdbscan"
    @param outlier_threshold: Threshold for detecting outliers. Can be in between 0 and 1.
    @param merge_edges_threshold: Threshold to clean up the graph after community detection, can be in between 0 and 1,
        edges greater than the threshold are merged.

    @param plot_interactive_graph:
    @param plot_labels: True or False to plot the ture labels and the assigned cluster labels.

    @return: The cluster labels create with this function.
    """

    # multi resolution meta graph
    multires_graph = mcc.multiresolution_graph(clustering_data, settings_data, "all",
                                               neighbour_based=connect_graph_neighbour_based)
    print("Multi-graph-build done, Time:", time.time() - start)

    # community detection
    if connect_graph_neighbour_based == True and outlier_mulit_res == "hdbscan" and community_mulit_res == "hdbscan":
        print("Neighbor based graph does not support interaction with HDBscan.")
        multires_graph = mcc.multires_community_detection(multires_graph, community_detection="leiden",
                                                          merge_edges_threshold=merge_edges_threshold,
                                                          outlier_detection="probability",
                                                          outlier_detection_threshold=outlier_threshold,
                                                          clustering_data=clustering_data)
    else:
        multires_graph = mcc.multires_community_detection(multires_graph, community_detection=community_mulit_res,
                                                          merge_edges_threshold=merge_edges_threshold,
                                                          outlier_detection_threshold=outlier_threshold,
                                                          outlier_detection=outlier_mulit_res,
                                                          clustering_data=clustering_data)
    print("Communities detected, Time:", time.time() - start)

    # create clustering labels and plot them if plot_labels == True
    df_clusters = mcc.graph_to_cell_labels_df(multires_graph)
    cluster_labels = mcc.df_cell_clusters_to_labels(df_clusters, adata_s2d1, plot_labels)

    # plot multi-graph with bokeh
    if plot_interactive_graph:
        mcc.interactive_plot(adata, clustering_data, multires_graph, create_upsetplot=False,
                             create_edge_weight_barchart=False, layout_option="hierarchy")

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

    # read adata file and select the set used for the clustering.
    adata = sc.read_h5ad(HERE / "data/cite/cite_gex_processed_training.h5ad")
    adata_s2d1 = adata[adata.obs.batch == "s2d1", :].copy()

    # simulated data
    #adata = sc.read_h5ad(HERE / "data/cite/sim-groups-contsclust.h5ad")
    #clustering_data = adata.uns["constclust"]['clusterings']
    #settings_data = adata.uns["constclust"]['settings']
    #clustering_data = clustering_data.reset_index()
    #clustering_data = clustering_data.rename({'index': 'cell'}, axis=1)

    # check if adata file has calculated the umap plot
    if "X_umap" not in adata.obsm_keys():
        sc.tl.umap(adata)

    print("Read data, Time:", time.time() - start)

    run_multires_consensus_clustering(clustering_data, settings_data, adata=adata_s2d1,
                                      community_mulit_res="leiden", merge_edges_threshold=0.8,
                                      outlier_mulit_res="probability", outlier_threshold=0.9,
                                      connect_graph_neighbour_based=True, plot_labels=False, plot_interactive_graph=False)


    #opti.work_on_saved_graphs()