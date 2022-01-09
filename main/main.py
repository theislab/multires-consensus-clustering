from multires_consensus_clustering import meta_graph as mg
import multires_consensus_clustering as mcc
from pathlib import Path
import time
import scanpy as sc
import igraph as ig

HERE = Path(__file__).parent.parent



def meta_graph():
    """
    Uses the Meta Graph script to build the graph from the sc data.

    @return: The meta graph; "graph" an igraph object graph.
    """

    # read data
    clustering_data = mg.read_data(HERE / "data\s2d1_clustering.tsv", "all")
    settings_data = mg.read_data(HERE / "data\s2d1_settings.tsv", "all")

    # binning the clusterings in bins close to the given numbers, combines all bins contained in the list
    number_of_clusters_data = mg.sort_by_number_clusters(settings_data, clustering_data, [4])

    # builds the graph from the bins
    graph = mg.build_graph(number_of_clusters_data, clustering_data)

    # get average edge weight
    mean_edge_value = mcc.plot_edge_weights(graph, plot_on_off=False)

    # find outliers using hdbscan
    graph = mcc.hdbscan_outlier(graph, mean_edge_value, plot_on_off=False)

    # delete all edges below threshold
    #graph = mcc.delete_edges_below_threshold(graph, mean_edge_value)

    # builds a consensus graph by taking different graphs clustering them and intersecting the clusters.
    #graph = mcc.consensus_graph(graph)

    # uses hdbscan for community detection
    #graph = mcc.hdbscan_community_detection(graph)

    # deletes outlier communities using normalized community size.
    #graph = mcc.delete_small_node_communities(graph)

    # detect and merge communities in the meta graph
    graph = mcc.igraph_community_detection(graph, detection_algorithm="louvain")

    # contract graph clustering into single node
    graph = mcc.contract_graph(graph)

    # create a pandas dataframe with the probabilities of a cell being in a node
    df_cell_probability = mcc.graph_nodes_cells_to_df(graph, clustering_data)

    # create a pandas dataframe with the final clustering labels
    mcc.assign_cluster_to_cell(df_cell_probability)

    return graph


def interactive_plot(graph, create_upsetplot, create_edge_weight_barchart):
    """
    Uses the adata to build an interactive plot with bokeh.

    @param graph: The graph on which the plot is based.
    @param create_upsetplot: Boolean variable to turn the upsetplot on or off; "True"/"False"
    @param create_edge_weight_barchart: Boolean variable to turn the edge weight barchart on or off; "True"/"False"
    """

    # read data
    clustering_data = mg.read_data(HERE / "data\s2d1_clustering.tsv", "all")
    settings_data = mg.read_data(HERE / "data\s2d1_settings.tsv", "all")

    # read adata file and select the set used for the clustering.
    adata = sc.read_h5ad(HERE / "data/cite/cite_gex_processed_training.h5ad")
    adata_s2d1 = adata[adata.obs.batch == "s2d1", :].copy()

    # creates a pandas df with the probabilities of a cell being in a specific node
    df_cell_probability = mcc.graph_nodes_cells_to_df(graph, clustering_data)

    # create an bar-chart with all edge weights
    if create_edge_weight_barchart:
        mcc.plot_edge_weights(graph)

    # create an upsetplot for the data
    if create_upsetplot:
        mcc.upsetplot_graph_nodes(df_cell_probability)

    # create the cluster plots from the adata and adds the images to the graph
    graph = mcc.umap_plot(df_cell_probability, adata_s2d1, graph)

    # plots an interactive graph using bokeh and an upset plot showing how the cells are distributed
    mcc.plot_interactive_graph(graph, df_cell_probability)


# run program
if __name__ == "__main__":
    start = time.time()

    meta_graph = meta_graph()
    #interactive_plot(meta_graph, create_upsetplot=False, create_edge_weight_barchart=False)

    # measure the time
    end = time.time()
    print("Time to run: ", end - start)
