import pytest
from multires_consensus_clustering import Meta_Graph as mg
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
from sklearn import metrics


clustering_data = mg.read_data(HERE / "toy_data_clustering.tsv", "all")
settings_data = mg.read_data(HERE / "toy_data_settings.tsv", "all")


def test_jaccard_index():
    """
    Test if the jaccard index is calculated correctly using the toy dataset and np.random.randint().
    """

    assert mg.clustering_edges_array(clustering_data["C001"].values, clustering_data["C002"].values) == [(0, 0, 0.125),
        (0, 1, 0.5), (1, 0, 0.2857142857142857), (1, 1, 0.3333333333333333), (2, 0, 0.25)]


    set_1 = np.random.randint(low=0,high=5,size=100)
    set_2 = np.random.randint(low=6,high=10,size=100)

    tuple_list = []
    for index_set in range(len(set_1)):
        tuple_list.append(str(set_1[index_set]) + str(set_2[index_set]))

    list_of_unique_tuples, counts_tuple = np.unique(tuple_list, return_counts=True)

    list_of_counts = []
    for unique_tuple in list_of_unique_tuples:
        unique_count = 0
        for all_tuple in tuple_list:
            if unique_tuple[0] in all_tuple or unique_tuple[1] in all_tuple:
                unique_count += 1
        list_of_counts.append(unique_count)

    edge_weight_jaccard_list = []
    for edge in mg.clustering_edges_array(set_1,set_2):
        edge_weight_jaccard_list.append(edge[2])

    assert edge_weight_jaccard_list == edge_weight_jaccard_list
def test_build_graph():
    """
    Test if the graph is build correctly using the toy dataset. Asses using the edgelist of the graph.
    """
    number_of_clusters_data = mg.sort_by_number_clusters(settings_data, clustering_data, 4)
    graph = mg.build_graph(number_of_clusters_data)
    assert graph.get_edgelist() == [(0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (0, 5), (0, 6), (0, 7), (3, 5), (3, 6),
        (3, 7), (4, 8), (0, 9), (0, 10), (0, 11), (3, 10), (3, 11), (4, 11), (0, 12), (0, 13), (3, 12), (3, 13),
        (4, 12), (1, 5), (1, 8), (2, 6), (2, 7), (1, 9), (1, 11), (2, 9), (2, 10), (2, 11), (1, 12), (1, 13), (2, 12),
        (2, 13), (5, 9), (5, 11), (6, 9), (6, 10), (6, 11), (7, 10), (7, 11), (8, 11), (5, 13), (6, 12), (6, 13),
        (7, 12), (7, 13), (8, 12), (9, 13), (10, 12), (10, 13), (11, 12), (11, 13)]


    number_of_clusters_data = mg.sort_by_number_clusters(settings_data, clustering_data, 2)
    graph = mg.build_graph(number_of_clusters_data)
    assert graph.vs.indices == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def test_binning():
    df_binning = mg.sort_by_number_clusters(settings_data, clustering_data, 2)
    df_control = clustering_data.loc[:, clustering_data.columns != 'cell']
    assert df_binning.equals(df_control)

    df_binning = mg.sort_by_number_clusters(settings_data, clustering_data, 4)
    df_control = clustering_data.loc[:, clustering_data.columns != 'cell']
    assert df_binning.equals(df_control)
