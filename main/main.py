import multires_consensus_clustering as mcc
from pathlib import Path
import time
import scanpy as sc
import pandas as pd
import igraph as ig
import numpy as np


HERE = Path(__file__).parent.parent


def run_multires_consensus_clustering(clustering_data, settings_data, adata, plot_edge_weights, plot_labels,
                                      plot_interactive_graph, combine_mulit_res, community_mulit_res,
                                      merge_edges_mulit_res, outlier_mulit_res, connect_graph_neighbour_based):
    """
    Run the multires consensus clustering by first building a meta_graph for every bin (resolution),
        run community detection on these graphs and merge similiar nodes.
        Create a graph based on all the merged resolution called multi_graph and
        run community detection again to create a merged graph tree representing the clustering resolutions.
    @param connect_graph_neighbour_based: True or False, connects the multi-resolution-graph either
        by neighbouring resolutions or connects all vertices. HDBscan does not work with the neighbour based graph!
    @param outlier_mulit_res: "probability" or "hdbscan"
    @param merge_edges_mulit_res: Threshold to clean up the graph after community detection, edges > threshold are merged
    @param community_mulit_res: "leiden", "hdbscan", "component" or otherwise automatically louvain.
    @param combine_mulit_res: "frist" or "list" combines the graph attributes by function.
    @param plot_interactive_graph: True or False to plot the interactive graph.
    @param plot_labels: True or False to plot the ture labels and the assigned cluster labels.
    @param plot_edge_weights: True or False to plot edge weight distribution.
    @param settings_data: The settings data of the clusterings, number of clusters parameters, etc.
    @param clustering_data: The clustering data based of the adata file, as a pandas dataframe, e.g. cell, C001, ...
    @param adata: The single cell adata file
    @return: The cluster labels create with this function.
    """

    # multi resolution meta graph
    multires_graph = mcc.multiresolution_graph(clustering_data, settings_data, "all",
                                               neighbour_based=connect_graph_neighbour_based)
    print("Multi-graph-build done, Time:", time.time() - start)
    mcc.write_graph_to_file(multires_graph, neighbour_based=False)

    # community detection
    if connect_graph_neighbour_based == True and outlier_mulit_res == "hdbscan" and community_mulit_res == "hdbscan":
        print("Neighbor based graph does not support interaction with HDBscan.")
        multires_graph = mcc.multires_community_detection(multires_graph, combine_by=combine_mulit_res,
                                                          community_detection="leiden",
                                                          merge_edges_threshold=merge_edges_mulit_res,
                                                          outlier_detection="probability",
                                                          clustering_data=clustering_data)
    else:
        multires_graph = mcc.multires_community_detection(multires_graph, combine_by=combine_mulit_res,
                                                      community_detection=community_mulit_res,
                                                      merge_edges_threshold=merge_edges_mulit_res,
                                                      outlier_detection=outlier_mulit_res, clustering_data=clustering_data)
    print("Communities detected, Time:", time.time() - start)

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
                                      plot_labels=True, plot_interactive_graph=False, combine_mulit_res="list",
                                      community_mulit_res="leiden", merge_edges_mulit_res=1,
                                      outlier_mulit_res="probability", connect_graph_neighbour_based=True)

