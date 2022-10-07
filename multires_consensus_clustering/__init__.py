"""
Code for the multi-resolution consensus clustering project.
"""
__version__ = "0.0.1"


from .consistency_cluster_labels import cluster_consistency
from .uncertainty_measure import uncertainty_measure_cells
from .binning import bin_n_clusters
from .meta_graph import build_graph, sort_by_number_clusters, meta_graph, read_data, \
    create_and_plot_single_resolution_graph
from .graph_analysis import create_distance_matrix, weighted_jaccard, jaccard_index_two_vertices, merge_by_list
from. graph_community_detection import igraph_community_detection, contract_graph, hdbscan_community_detection, \
    component_merger
from .interactive_plot import plot_interactive_graph, interactive_plot, umap_plot
from .merge_nodes import merge_nodes, merge_edges_weight_above_threshold, merge_list_of_sets
from .cell_labels_scanpy import relabel_cell, true_labels, graph_to_cell_labels_df, df_cell_clusters_to_labels
from .outlier_detection import delete_small_node_communities, delete_edges_below_threshold, \
    hdbscan_outlier, delete_nodes_with_zero_degree, filter_by_node_probability
from .merge_resolution_graphs import merge_two_resolution_graphs, delete_edges_single_resolution, reconnect_graph, \
    multires_community_detection, multiresolution_graph
from .save_and_load_graphs import write_graph_to_file, load_graph_from_file
from .cell_probability import graph_nodes_cells_to_df, single_node_to_df
from .informativ_plots import plot_edge_weights, upsetplot_graph_nodes, cell_occurrence_plot, \
    vertex_probabilities_plot, probability_umap_plot, node_level_plot
