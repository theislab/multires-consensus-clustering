"""
Code for the multi-resolution consensus clustering project.
"""
__version__ = "0.0.1"

from .binning import bin_n_clusters
from .Meta_Graph import build_graph, plot_graph
from .graph_analysis import min_cuts, graph_community_detection
from .interactive_plot import plot_interactive_graph, bar_chart_nodes
from .merge_nodes import merge_nodes, merge_edges_weight_1
