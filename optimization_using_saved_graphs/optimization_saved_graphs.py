import multires_consensus_clustering as mcc
from pathlib import Path
import time
import numpy as np
import scanpy as sc
import pandas as pd
import igraph as ig

HERE = Path(__file__).parent.parent


def overall_assignment_probability(graph):
    """
    Calculates the overall probability as a measurement of node certainty.
    @param graph: iGraph graph, need attribute vs["probability_df"].
    @return: The overall probability score.
    """
    overall_prob = 0
    for vertex in graph.vs:
        probabilities = vertex["probability_df"]
        overall_prob += sum(probabilities) / np.count_nonzero(probabilities)

    overall_prob = overall_prob / graph.vcount()

    return overall_prob


# work on the saved graphs for optimization
def work_on_save_graphs(adata_s2d1, clustering_data, settings_data, plot_labels, plot_interactive_graph,
                        community_detection, combine_by, merge_edges):
    """
    Function for working on the saved graphs. Testing ground for different approaches on the multi-resolution graph.
    """
    start = time.time()

    # load the graph from pickle file
    multires_graph = mcc.load_graph_from_file(neighbour_based=False)
    print("Load graph from pickle-file, Time:", time.time() - start)

    # outlier detection
    # multires_graph = mcc.hdbscan_outlier(graph=multires_graph, threshold=threshold, plot_on_off=False)
    multires_graph = mcc.filter_by_node_probability(multires_graph, 0.5)
    multires_graph = mcc.merge_edges_weight_above_threshold(multires_graph, threshold=1)

    # resolution parameters from the clustering files
    resolutions_settings = sum(settings_data["resolution"]) / len(settings_data["resolution"])
    resolution_n_clusters = sum(settings_data["n_clusters"]) / len(settings_data["n_clusters"])


    if community_detection == "resolution_based":
        clustering_graph = ig.Graph.community_leiden(multires_graph, weights="weight", objective_function="modularity",
                                                     n_iterations=2, resolution_parameter=resolution_n_clusters)

    elif community_detection == "hdbscan":
        clustering_graph = mcc.hdbscan_community_detection(multires_graph)
    else:
        clustering_graph = ig.Graph.community_leiden(multires_graph, weights="weight", objective_function="CPM")

    print("Communities detected, Time:", time.time() - start)

    # combine attributes by list or first
    if combine_by == "list":
        graph = mcc.merge_by_list(clustering_graph)
    else:
        graph = clustering_graph.cluster_graph(combine_vertices="first", combine_edges=max)

    # delete edges and reconnect the graph
    graph.delete_edges()
    graph = mcc.reconnect_graph(graph)

    # clean up the graph with edge merger
    multires_graph = mcc.merge_edges_weight_above_threshold(graph, threshold=merge_edges)

    print("Probability per node:", overall_assignment_probability(multires_graph))

    # plot cluster labels
    if plot_labels:
        # cluster_labels = mcc.graph_to_clustering(multires_graph, adata_s2d1, cluster_or_probability="cluster")
        true_labels = mcc.true_labels(labels_df=mcc.read_data(HERE / "data\s2d1_labels.tsv", "all"), adata=adata_s2d1)
        cluster_labels = mcc.best_prob_cell_labels(multires_graph, adata=adata_s2d1)

    # plot interactive graph
    if plot_interactive_graph:
        mcc.interactive_plot(adata_s2d1, clustering_data, multires_graph, create_upsetplot=False,
                             create_edge_weight_barchart=False, layout_option="hierarchy")
