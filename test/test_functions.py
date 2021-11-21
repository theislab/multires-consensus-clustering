import pytest
from multires_consensus_clustering import Meta_Graph as mg
from pathlib import Path
HERE = Path(__file__).parent



clustering_data = mg.read_data(HERE / "toy_data_clustering.tsv", "all")
settings_data = mg.read_data(HERE / "toy_data_settings.tsv", "all")


def test_jaccard_index():
    """
    Test if the jaccard index is calculated correctly using the toy dataset.
    """
    assert mg.clustering_edges_array(clustering_data["C001"].values ,clustering_data["C002"].values) == [(0, 0, 0.125),
        (0, 1, 0.5), (1, 0, 0.2857142857142857), (1, 1, 0.3333333333333333), (2, 0, 0.25)]

    assert mg.clustering_edges_array(clustering_data["C001"].values ,clustering_data["C003"].values) == [(0, 0, 0.14285714285714285),
        (0, 1, 0.5), (0, 2, 0.14285714285714285), (1, 0, 0.3333333333333333), (1, 1, 0.125), (1, 2, 0.3333333333333333), (2, 3, 1.0)]

def test_build_graph():
    """
    Test if the graph is build correctly using the toy dataset. Asses using the edgelist of the graph.
    """
    number_of_clusters_data = mg.sort_by_number_clusters(settings_data, clustering_data, 4)
    graph = mg.build_graph(number_of_clusters_data)
    assert graph.get_edgelist() == [(0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (0, 5), (0, 6), (0, 7), (3, 5), (3, 6),
        (3, 7), (4, 8), (0, 9), (0, 10), (0, 11), (3, 10), (3, 11), (4, 11), (1, 5), (1, 8), (2, 6), (2, 7), (1, 9),
        (1, 11), (2, 9), (2, 10), (2, 11), (5, 9), (5, 11), (6, 9), (6, 10), (6, 11), (7, 10), (7, 11), (8, 11)]


    number_of_clusters_data = mg.sort_by_number_clusters(settings_data, clustering_data, 2)
    graph = mg.build_graph(number_of_clusters_data)
    assert graph.get_edgelist() == [(0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (0, 5), (0, 6), (0, 7), (3, 5), (3, 6),
                                    (3, 7), (4, 8), (0, 9), (0, 10), (0, 11), (3, 10), (3, 11), (4, 11), (1, 5), (1, 8),
                                    (2, 6), (2, 7), (1, 9),
                                    (1, 11), (2, 9), (2, 10), (2, 11), (5, 9), (5, 11), (6, 9), (6, 10), (6, 11),
                                    (7, 10), (7, 11), (8, 11)]

def test_binning():
    df_binning = mg.sort_by_number_clusters(settings_data, clustering_data, 2)
    df_control = clustering_data
    print(df_binning)
    print(df_control)
    assert df_binning.equals(df_control)

    df_binning = mg.sort_by_number_clusters(settings_data, clustering_data, 4)
    df_control = clustering_data
    assert df_binning.equals(df_control)

test_jaccard_index()
test_build_graph()
test_binning()
