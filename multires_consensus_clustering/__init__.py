"""
Code for the multi-resolution consensus clustering project.
"""
__version__ = "0.0.1"

from .binning import bin_n_clusters
from .Meta_Graph import build_graph, plot_graph
from .graph_analysis import merg_edges_weight_1, min_cuts, graph_community_detection
from .interactive_plot import plot_interactive_graph
