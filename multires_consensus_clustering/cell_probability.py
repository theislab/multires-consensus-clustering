import pandas
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path


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

    # create df for the probability values
    graph_df = pd.DataFrame()
    graph_df["cell"] = clustering_data["cell"]
    graph_df.set_index("cell", inplace=True)

    vertex_index = 0
    for merged_node in label_list:
        # count cells
        cell_names_vertex, counts = np.unique(merged_node, return_counts=True)
        # calculate the probability
        counts = counts / len(graph.vs[vertex_index]["name"])

        # creates new column in df
        df_probabilities_vertex = pd.DataFrame({'cell': cell_names_vertex,
                            'merged_node_' + str(vertex_index): counts})
        df_probabilities_vertex.set_index("cell", inplace=True)

        # assigns the values to the cells
        graph_df = pd.concat([graph_df, df_probabilities_vertex], axis=1)

        vertex_index +=1

    # change NaN values to 0
    graph_df = graph_df.fillna(0)

    graph.vs["cell_index"] = [clustering_data["cell"].values] * graph.vcount()

    return graph_df


def single_node_to_df(vertex, clustering_data):
    """
    Create as pandas dataframe with the probability of each cell occurring in the given node.

    @param vertex: iGraph vertex (node), needs attribute ["cell"]
    @param clustering_data: The clustering data, type pandas.dataframe, e.g. cell, C001, C002, ...
    @return: Return a pandas dataframe with cells as index and probability sorted by node.
    """
    # create list with all cell names in vertex
    cluster_cell_list = sum(vertex["cell"], [])

    # create df for the probability values
    graph_df = pd.DataFrame()
    graph_df["cell"] = clustering_data["cell"]
    graph_df.set_index("cell", inplace=True)

    # count cells
    cell_names_vertex, counts = np.unique(cluster_cell_list, return_counts=True)
    # calculate the probability
    counts = counts / len(vertex["name"])

    # creates new column in df
    df_probabilities_vertex = pd.DataFrame({'cell': cell_names_vertex,
                                            'merged_node': counts})
    df_probabilities_vertex.set_index("cell", inplace=True)

    # assigns the values to the cells
    graph_df = pd.concat([graph_df, df_probabilities_vertex], axis=1)

    # change NaN values to 0
    graph_df = graph_df.fillna(0)

    vertex["cell_index"] = clustering_data["cell"]

    return graph_df
