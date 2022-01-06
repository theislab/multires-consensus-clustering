import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import base64
from io import BytesIO
import imageio
import matplotlib.pyplot as plt

HERE = Path(__file__).parent.parent


def graph_nodes_cells_to_df(graph, data):
    """
    Create as pandas dataframe with the probability of each cell occurring in one node.

    @param graph: The graph, need attribute graph.vs["cell"]
    @param data: The clustering data, type pandas.dataframe, e.g. cell, C001, C002, ...
    @return: Return a pandas dataframe with cells as index and probability sorted by node.
    """
    label_list = []

    for merged_vertices in graph.vs["cell"]:
        cluster_cell_list = []
        for cell_list in merged_vertices:
            cluster_cell_list = cell_list + cluster_cell_list
        label_list.append(cluster_cell_list)

    cell_names = pd.DataFrame(data["cell"])

    i = 0
    list_df_nodes = []

    for merged_node in label_list:
        values, counts = np.unique(merged_node, return_counts=True)
        counts = counts / len(graph.vs[i]["name"])
        node_cell_count = pd.DataFrame({"cell": values, 'merged_node_' + str(i): counts})
        merged_cell_count = pd.concat([cell_names.set_index("cell"), node_cell_count.set_index("cell")], axis=1)
        list_df_nodes.append(merged_cell_count)

        i += 1

    all_cell_counts = pd.DataFrame()
    for cell_df in list_df_nodes:
        all_cell_counts = pd.concat([all_cell_counts, cell_df])

    all_cell_counts = all_cell_counts.groupby(['cell']).sum()

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
    Uses the umap from scanpy to plat the probability of the cells being in one node. These range from 0 to 1.
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


def assing_cells_to_cluster(df_cell_probability):
    """

    @return: df_cell_clusters
    """
    df_cell_clusters = df_cell_probability.iloc[:, 0]
    for column in df_cell_probability.columns:
        for cell_probability in df_cell_probability[column].values:
            if cell_probability > df_cell_clusters.iloc[cell_probability.index, column]:
                print(test)


