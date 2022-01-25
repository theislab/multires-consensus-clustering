from multires_consensus_clustering import meta_graph as mg
import multires_consensus_clustering as mcc
from pathlib import Path
import time
import scanpy as sc
import pandas as pd
import igraph as ig

HERE = Path(__file__).parent.parent


def meta_graph(clustering_data, settings_data, bin):
    """
    Uses the Meta Graph script to build the graph from the sc data.

    @param clustering_data:
    @param settings_data:
    @param bin:
    @return: The meta graph; "graph" an igraph object graph.
    """

    # binning the clusterings in bins close to the given numbers, combines all bins contained in the list
    number_of_clusters_data = mg.sort_by_number_clusters(settings_data, clustering_data, bin)

    # builds the graph from the bins
    graph = mg.build_graph(number_of_clusters_data, clustering_data)

    # delete all edges below threshold
    #graph = mcc.delete_edges_below_threshold(graph, 0.5)

    # get average edge weight
    #mean_edge_value = mcc.plot_edge_weights(graph, plot_on_off=False)

    # delete all nodes with zero degree
    # graph = mcc.delete_nodes_with_zero_degree(graph)

    # find outliers using hdbscan
    #graph = mcc.hdbscan_outlier(graph, mean_edge_value, plot_on_off=False)

    # builds a consensus graph by taking different graphs clustering them and intersecting the clusters.
    # graph = mcc.consensus_graph(graph)

    # deletes outlier communities using normalized community size.
    # graph = mcc.delete_small_node_communities(graph)

    # uses hdbscan for community detection
    # graph = mcc.hdbscan_community_detection(graph)

    # detect and merge communities in the meta graph
    graph = mcc.igraph_community_detection(graph, detection_algorithm="louvain")

    # ig.plot(graph)

    # contract graph clustering into single node
    graph = mcc.contract_graph(graph)

    return graph


def multiresolution_graph(clustering_data, settings_data, list_resolutions, neighbour_based):
    """
    Creates a multi-resolution graph based on the resolutions given in the list_resolutions.
    Can either create a graph where all resolution vertices are connected or only the neighbouring resolutions are connected.

    @param settings_data: Settings data about the clusters.
    @param clustering_data: The cluster data.
    @param neighbour_based: Boolean to decided on the way to connect the vertices across resolutions.
    @param list_resolutions: The list containing the different resolutions, e.g. [3,5,9,20, ... ] or "all"
    @return: The mulit-graph as a iGraph graph.
    """

    # set maximum level for hierarchy plot
    level_count = len(list_resolutions) + 1

    # check if list resolution contains less the two resolutions
    if len(list_resolutions) <= 1:
        print("More then one resolution needed for multi resolution graph.")

        return meta_graph(clustering_data, settings_data, list_resolutions[0])

    # create the multi graph
    else:
        if list_resolutions == "all":
            bins_clusterings = mcc.bin_n_clusters(settings_data["n_clusters"])
            list_resolutions = [int(first_number_clusters[0]) for first_number_clusters in bins_clusterings]
        else:
            # sort resolution list in cases not sorted
            list_resolutions.sort()

        # create first graph and assign the level
        resolution_1 = meta_graph(clustering_data, settings_data, list_resolutions[0])
        resolution_1.vs["level"] = [level_count] * resolution_1.vcount()

        # create new attribute to save the cell probabilities in a meta node
        probability_df = mcc.graph_nodes_cells_to_df(resolution_1, clustering_data)
        resolution_1.vs["probability_df"] = [probability_df[column].values for column in probability_df.columns]

        # delete all edges of the old graph
        mcc.delete_edges_single_resolution(resolution_1)

        # change level count
        level_resolution_2 = level_count - 1

        # create multi-graph using the rest of the list_resolutions
        for resolution in list_resolutions[1:]:

            # create graph and assign the level
            resolution_2 = meta_graph(clustering_data, settings_data, resolution)
            resolution_2.vs["level"] = [level_resolution_2] * resolution_2.vcount()

            # delete all edges of the old graph
            mcc.delete_edges_single_resolution(resolution_2)

            # create multi graph based on neighbours or connect all vertices
            if neighbour_based:
                resolution_1 = mcc.merge_two_resolution_graphs(resolution_1, resolution_2,
                                                               current_level=level_resolution_2 + 1,
                                                               neighbours=True, clustering_data=clustering_data)
            else:
                # connect all vertices
                resolution_1 = mcc.merge_two_resolution_graphs(resolution_1, resolution_2, current_level=None,
                                                               neighbours=False, clustering_data=clustering_data)

            # set level for next graph
            level_resolution_2 -= 1

        # return the final multi-graph
        return resolution_1


def interactive_plot(graph, create_upsetplot, create_edge_weight_barchart, layout_option):
    """
    Uses the adata to build an interactive plot with bokeh.

    @param layout_option: The layout for the graph, can be "hierarchy" -> layout based on vertex level.
        Else: iGraph auto layout
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
        mcc.plot_edge_weights(graph, plot_on_off=True)

    # create an upsetplot for the data
    if create_upsetplot:
        mcc.upsetplot_graph_nodes(df_cell_probability)

    # create the cluster plots from the adata and adds the images to the graph
    graph = mcc.umap_plot(df_cell_probability, adata_s2d1, graph)

    # plots an interactive graph using bokeh and an upset plot showing how the cells are distributed
    mcc.plot_interactive_graph(graph, df_cell_probability, layout_option)


# run program
if __name__ == "__main__":
    start = time.time()

    # read data
    clustering_data = mg.read_data(HERE / "data\s2d1_clustering.tsv", "all")
    settings_data = mg.read_data(HERE / "data\s2d1_settings.tsv", "all")

    # single resolution meta graph
    # meta_graph = meta_graph(clustering_data, settings_data, [4])
    # interactive_plot(meta_graph, create_upsetplot=False, create_edge_weight_barchart=False, graph_hierarchy="auto")

    # multi resolution meta graph
    multires_graph = multiresolution_graph(clustering_data, settings_data, "all", neighbour_based=True)
    multires_graph = mcc.delete_edges_below_threshold(multires_graph, 0.5)
    interactive_plot(multires_graph, create_upsetplot=False, create_edge_weight_barchart=False, layout_option="hierarchy")

    # measure the time
    end = time.time()
    print("Time to run: ", end - start)
