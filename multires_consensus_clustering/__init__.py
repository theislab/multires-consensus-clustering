"""
Code for the multi-resolution consensus clustering project.
"""
__version__ = "0.0.1"

from .binning import bin_n_clusters
from .meta_graph import build_graph, sort_by_number_clusters, meta_graph, read_data, \
    create_and_plot_single_resolution_graph
from .graph_analysis import igraph_community_detection, plot_edge_weights, \
    consensus_graph, contract_graph, hdbscan_community_detection, create_distance_matrix, weighted_jaccard, \
    jaccard_index_two_vertices, merge_by_list
from .interactive_plot import plot_interactive_graph, upsetplot_graph_nodes, interactive_plot, umap_plot
from .merge_nodes import merge_nodes, merge_edges_weight_above_threshold
from .cell_labels_scanpy import relabel_cell, graph_nodes_cells_to_df, assign_cluster_to_cell, \
    single_node_to_df, graph_to_clustering, true_labels, best_prob_cell_labels
from .outlier_detection import delete_small_node_communities, delete_edges_below_threshold, \
    hdbscan_outlier, delete_nodes_with_zero_degree, filter_by_node_probability
from .merge_resolution_graphs import merge_two_resolution_graphs, delete_edges_single_resolution, reconnect_graph, \
    component_merger, multires_community_detection, multiresolution_graph
from .save_and_load_graphs import write_graph_to_file, load_graph_from_file
