"""
Code for the multi-resolution consensus clustering project.
"""
__version__ = "0.0.1"

from .binning import bin_n_clusters
from .meta_graph import build_graph, plot_graph
from .graph_analysis import igraph_community_detection, plot_edge_weights, \
    consensus_graph, contract_graph, hdbscan_community_detection, create_distance_matrix
from .interactive_plot import plot_interactive_graph, upsetplot_graph_nodes
from .merge_nodes import merge_nodes, merge_edges_weight_1
from .cell_labels_scanpy import relabel_cell, graph_nodes_cells_to_df, umap_plot, assign_cluster_to_cell
from .outlier_detection import min_cuts, delete_small_node_communities, delete_edges_below_threshold, \
    hdbscan_outlier, delete_nodes_with_zero_degree
