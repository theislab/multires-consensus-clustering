import pandas
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import base64
from io import BytesIO
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt

HERE = Path(__file__).parent.parent


def graph_nodes_cells_to_df(graph, clustering_data):
    """
    Create as pandas dataframe with the probability of each cell occurring in one node.

    @param graph: The graph, need attribute graph.vs["cell"]
    @param clustering_data: The clustering data, type pandas.dataframe, e.g. cell, C001, C002, ...
    @return: Return a pandas dataframe with cells as index and probability sorted by node.
    """
    label_list = []

    for merged_vertices in graph.vs["cell"]:
        label_list.append(sum(merged_vertices, []))

    i = 0
    list_df_nodes = []

    for merged_node in label_list:
        # count cells
        values, counts = np.unique(merged_node, return_counts=True)
        # calculate the probability
        counts = counts / len(graph.vs[i]["name"])
        # create a df
        list_df_nodes.append(pd.DataFrame({"cell": values, 'merged_node_' + str(i): counts}))

        i += 1

    # merge all node df into a single dataframe
    all_cell_counts = pd.DataFrame()
    for cell_df in list_df_nodes:
        all_cell_counts = pd.concat([all_cell_counts, cell_df])

    # replace NaN values with 0
    all_cell_counts.fillna(0)

    all_cell_counts = all_cell_counts.groupby(['cell']).sum()
    return all_cell_counts


def single_node_to_df(vertex, clustering_data):
    """
    Create as pandas dataframe with the probability of each cell occurring in the given node.

    @param vertex: iGraph vertex (node), needs attribute ["cell"]
    @param clustering_data: The clustering data, type pandas.dataframe, e.g. cell, C001, C002, ...
    @return: Return a pandas dataframe with cells as index and probability sorted by node.
    """
    cluster_cell_list = []
    for cell_list in vertex["cell"]:
        cluster_cell_list = cell_list + cluster_cell_list

    cell_names = pd.DataFrame(clustering_data["cell"])

    values, counts = np.unique(cluster_cell_list, return_counts=True)
    counts = counts / len(vertex["name"])
    node_cell_count = pd.DataFrame({"cell": values, 'merged_node': counts})
    merged_cell_count = pd.concat([cell_names.set_index("cell"), node_cell_count.set_index("cell")], axis=1)

    all_cell_counts = merged_cell_count.groupby(['cell']).sum()

    return all_cell_counts


def relabel_cell(df_cell_probability, adata_s2d1):
    """
    Assigns all cell labels from the clustering the original cell_type from the adata (TCATTCAGTCACCGAC-1-s2d1 -> Erythroblast).
    If cell labels from clustering is not in adata, assigns "unknown".

    @param df_cell_probability: Pandas dataframe with cells as rows, merged nodes as clusters and
        values the probability of cell being in the node (0 to 1).
    @param adata_s2d1: The adata set on which the clustering is based.
    @return: Pandas dataframe with cell_types as rows and a summed up probability as values (weighted count by probability).
    """

    cell_type_dic = {}
    cell_index_counter = 0
    cell_index = adata_s2d1.obs["cell_type"].index

    print("len adata_s2d1 cell_type: ", len(adata_s2d1.obs["cell_type"]))
    print("len dataframe multires-consensus-clustering: ", len(df_cell_probability))

    labels_not_in_adata = set(df_cell_probability.index) - set(adata_s2d1.obs["cell_type"].index)

    for cell_name in adata_s2d1.obs["cell_type"]:
        cell_type_dic[cell_index[cell_index_counter]] = cell_name
        cell_index_counter += 1

    for cell_name in labels_not_in_adata:
        cell_type_dic[cell_name] = "unknown"

    df_cell_probability = df_cell_probability.rename(cell_type_dic)

    df_cell_probability = df_cell_probability.groupby(['cell']).sum()

    # pd.set_option('display.max_rows', None)
    print(df_cell_probability)

    return df_cell_probability


def umap_plot(df_cell_probability, adata, graph):
    """
    Uses the umap from scanpy to plot the probability of the cells being in one node. These range from 0 to 1.
    Saves the plot in directory plots und the node_names.

    @param df_cell_probability: Pandas dataframe with all cells from the clustering (as rows) and merged nodes as columns,
        values of the dataframe are the probabilities of the cell being in the merged node, range 0 to 1.
    @param adata: The original adata set, used for the layout of the plot.
    """

    # creates a plot for every merged node in the dataframe and adds the plot to the graph under G.vs["img"]
    # uses the code from Isaac for the encoding of the image
    # https://github.com/ivirshup/constclust/blob/6b7ef2773a3332beccd1de8774b16f3727321510/constclust/clustree.py#L223
    plot_list = []
    for columns in df_cell_probability.columns:
        adata.obs['probability_cell_in_node'] = df_cell_probability[columns]
        file = columns + ".png"
        plot = sc.pl.umap(adata, color='probability_cell_in_node', show=False)
        with BytesIO() as buf:
            plot.figure.savefig(buf, format="png", dpi=50)
            buf.seek(0)
            byte_image = base64.b64encode(buf.read())
        encode_image = byte_image.decode("utf-8")
        plot_list.append(f'<img src="data:image/png;base64,{encode_image}"/>')

    graph.vs["img"] = plot_list
    return graph


def assign_cluster_to_cell(df_cell_probability):
    """
    Returns clustering labels based on the highest probability.
    Creates a pandas dataframe with cells names as index and clustering labels in the column.

    @return df_cell_clusters: Pandas dataframe with cell names as index and clustering labels as column.
    """

    df_cell_clusters = pd.DataFrame({
        'cell': df_cell_probability.index,
        'probability': pd.Series([-1] * len(df_cell_probability), dtype='float'),
        'cluster_labels': [str(0)] * len(df_cell_probability)
    })

    df_cell_clusters['probability'].astype(dtype='float64')

    df_cell_clusters.set_index('cell', inplace=True, drop=True)

    cluster_label = 1
    for column in df_cell_probability.columns:
        index_row = 0
        for cell_probability in df_cell_probability[column]:
            if df_cell_clusters["probability"][index_row] < cell_probability and 0 < cell_probability:
                df_cell_clusters.iat[index_row, 0] = cell_probability
                df_cell_clusters.iat[index_row, 1] = str(cluster_label)
            index_row += 1
        cluster_label += 1


    print("Certainty cluster labels:", df_cell_clusters["probability"].sum() / len(df_cell_clusters))

    return df_cell_clusters


def graph_to_clustering(graph, adata, cluster_or_probability):
    """
    Creates cluster labels for the adata set using the df in each node of the graph and
        creates a static UMAP plot colored by cluster.

    @param cluster_or_probability: Plot either the cluster labels or the probabilities; "cluster" or "probability"
    @param adata: The cell data on which the graph is based.
    @param graph: The graph on which the clusters are based, iGraph graph. Vertices need the attribute "probability_df",
        graph.vs["probability_df"].
    @return: Returns the clustering labels as a pandas dataframe.
    """

    # check if "probability_df" is an attribute of the graph vertices
    if "probability_df" in graph.vs.attributes():
        selected_resolution_nodes = select_resolution_nodes(graph, resolution_level=-1)
        graph_clustering_df = pandas.DataFrame()
        graph_clustering_df["cell"] = graph.vs[0]["cell_index"]
        i = 0

        # create a dataframe with the probabilities of all nodes
        for vertex in selected_resolution_nodes:
            graph_clustering_df["vertex_end_of_tree" + "_" + str(i)] = vertex["probability_df"]
            i += 1
        graph_clustering_df.set_index("cell", inplace=True)

        # generate a single dataframe with probabilities and cluster labels
        clustering_labels = assign_cluster_to_cell(graph_clustering_df)

        # plot data
        # plot clusters
        if cluster_or_probability == "cluster":
            adata.obs["meta_clusters"] = clustering_labels["cluster_labels"]

            number_clusters = np.unique(clustering_labels["cluster_labels"].values)
            print("The graph has ", len(number_clusters) + 1, "clusters")

            plot = sc.pl.umap(adata, color=["meta_clusters"], show=True)

        # plot probabilities
        elif cluster_or_probability == "probability":
            adata.obs["meta_clusters"] = clustering_labels["probability"]

            plot = sc.pl.umap(adata, color=["meta_clusters"], show=True)

        return clustering_labels


def select_resolution_nodes(graph, resolution_level):
    """
    Selects nodes based on the lowest branch, second lowest, etc. And returns all such nodes in  a list.

    @param resolution_level: Which level should be selected. The lowest level is 0, the second lowest -1, etc.
    @param graph: The graph, has to be a tree structure where a node has only one edge from a higher node, iGraph graph.
    @return: list_selected_nodes, a list of all the nodes within the lowest, etc. part of the branch.
    """
    list_selected_nodes = []

    for vertex in graph.vs:
        all_edges_vertex = vertex.all_edges()
        number_edge_vertex = len(all_edges_vertex)

        if resolution_level == 0:
            if number_edge_vertex == 1:
                # the end of a branch (of the tree) has only one edge in/out
                # -> the target of the edge is the lowest node of the branch
               list_selected_nodes.append(graph.vs[all_edges_vertex[0].target])

        elif resolution_level == -1:
            # chooses the source of the edge instead of the target to get the second lowest
            if number_edge_vertex == 1:
                list_selected_nodes.append(graph.vs[all_edges_vertex[0].source])

    return list_selected_nodes


def true_labels(labels_df, adata):
    """

    @param labels_df:
    @param adata:
    @return:
    """

    label_dict = dict(zip(labels_df["cell"], labels_df["cell_type"]))
    new_labels = [label_dict[cell] for cell in list(adata.obs.index)]

    adata.obs["true_labels"] = new_labels

    print("The graph has ", len(np.unique(labels_df["cell_type"])) + 1, "cell types")

    plot = sc.pl.umap(adata, color=["true_labels"], show=True)

