import time

import pytest
import multires_consensus_clustering as mcc
from multires_consensus_clustering import merge_nodes
from pathlib import Path
import numpy as np
import igraph
import uuid

HERE = Path(__file__).parent
from sklearn import metrics

clustering_data = mcc.read_data(HERE / "toy_data_clustering.tsv")
settings_data = mcc.read_data(HERE / "toy_data_settings.tsv")


def test_jaccard_index():
    """
    Test if the jaccard index is calculated correctly using the toy dataset and np.random.randint().
    """

    assert mcc.clustering_edges_array(clustering_data["C001"].values, clustering_data["C002"].values) == [(0, 0, 0.125),
                                                                                                         (0, 1, 0.5), (
                                                                                                         1, 0,
                                                                                                         0.2857142857142857),
                                                                                                         (1, 1,
                                                                                                          0.3333333333333333),
                                                                                                         (2, 0, 0.25)]

    set_1 = np.random.randint(low=0, high=5, size=100)
    set_2 = np.random.randint(low=6, high=10, size=100)

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
    for edge in mcc.clustering_edges_array(set_1, set_2):
        edge_weight_jaccard_list.append(edge[2])

    assert edge_weight_jaccard_list == edge_weight_jaccard_list


def test_build_graph():
    """
    Test if the graph is build correctly using the toy dataset. Asses using the edgelist of the graph.
    """
    number_of_clusters_data = mcc.sort_by_number_clusters(settings_data, clustering_data, 4)
    graph = mcc.build_graph(number_of_clusters_data, clustering_data)

    # assert name are assigned correctly
    assert len(graph.vs["name"]) == len(['C001 : 0', 'C001 : 1', 'C001 : 2', 'C002 : 0', 'C002 : 1', 'C003 : 0', 'C003 : 1',
                                'C003 : 2', 'C003 : 3', 'C004 : 0', 'C004 : 1', 'C004 : 2', 'C005 : 0', 'C005 : 1'])

    assert sorted(graph.vs["name"]) == sorted(['C001 : 0', 'C001 : 1', 'C001 : 2', 'C002 : 0', 'C002 : 1', 'C003 : 0', 'C003 : 1',
                                'C003 : 2', 'C003 : 3', 'C004 : 0', 'C004 : 1', 'C004 : 2', 'C005 : 0', 'C005 : 1'])

    assert graph.get_edgelist() == [(0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (0, 5), (0, 6), (0, 7), (3, 5), (3, 6), (3, 7), (4, 8), (0, 9), (0, 10), (0, 11), (3, 10), (3, 11), (4, 11), (0, 12), (0, 13), (3, 12), (3, 13), (4, 12), (1, 5), (1, 8), (2, 6), (2, 7), (1, 9), (1, 11), (2, 9), (2, 10), (2, 11), (1, 12), (1, 13), (2, 12), (2, 13), (5, 9), (5, 11), (6, 9), (6, 10), (6, 11), (7, 10), (7, 11), (8, 11), (5, 13), (6, 12), (6, 13), (7, 12), (7, 13), (8, 12), (9, 13), (10, 12), (10, 13), (11, 12), (11, 13)]

    assert graph.es["weight"] == [0.125, 0.5, 0.28571, 0.33333, 0.25, 0.14286, 0.5, 0.14286, 0.33333, 0.125, 0.33333, 1.0, 0.4, 0.14286, 0.22222, 0.33333, 0.375, 0.16667, 0.25, 0.375, 0.25, 0.375, 0.2, 0.75, 0.25, 0.57143, 0.42857, 0.2, 0.42857, 0.125, 0.42857, 0.3, 0.125, 0.42857, 0.5, 0.3, 0.25, 0.28571, 0.2, 0.16667, 0.25, 0.5, 0.125, 0.16667, 0.5, 0.28571, 0.25, 0.33333, 0.125, 0.2, 0.33333, 0.14286, 0.28571, 0.57143, 0.2]

    # assert number of vertices are correct
    assert graph.vs.indices == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def test_binning():
    """
        Test if the bins of clusters are correctly calculated.

    """
    df_binning = mcc.sort_by_number_clusters(settings_data, clustering_data, 2)
    df_control = clustering_data.loc[:, clustering_data.columns != 'cell']
    assert df_binning.equals(df_control)

    df_binning = mcc.sort_by_number_clusters(settings_data, clustering_data, 4)
    df_control = clustering_data.loc[:, clustering_data.columns != 'cell']
    assert df_binning.equals(df_control)


def test_merge_nodes():
    """
        Test for node merging. Tests if all attributes are correctly added to the merged node
        and if the nodes are correctly merged
    """
    # create random graph
    graph = igraph.Graph.GRG(n=10, radius=2)
    number_nodes_start = graph.vcount()

    name = []
    clustering = []
    cell = []
    for i in graph.vs.indices:
        name.append(str(uuid.uuid4())[0:6] + str(i))
        clustering.append(str(uuid.uuid4())[0:6] + str(i))
        cell.append([str(uuid.uuid4())[0:6] + str(i), str(uuid.uuid4())[0:6] + str(i), str(uuid.uuid4())[0:6] + str(i)])

    graph.vs["name"] = name
    graph.vs["clustering"] = clustering
    graph.vs["cell"] = cell

    # merges node_0 and node_1
    graph_merged = merge_nodes(graph, 0, 1)

    # test if all attributes are merged correctly
    assert graph_merged.vs["name"][0] == [name[0], name[1]]

    assert graph_merged.vs["clustering"][0] == [clustering[0], clustering[1]]

    assert graph_merged.vs["cell"][0] == sum([cell[0], cell[1]], [])

    # merges node_1 (former node_1 and node_2 merged and reindex) and node_2 (former node_3)
    graph_merged = merge_nodes(graph, 0, 1)

    # test if attributes af merged nodes and new node are merged correctly
    assert graph_merged.vs["name"][0] == [name[0], name[1], name[2]]

    assert graph_merged.vs["clustering"][0] == [clustering[0], clustering[1], clustering[2]]

    assert graph_merged.vs["cell"][0] == sum([cell[0], cell[1], cell[2]], [])

    # tests if the number of nodes is reduced by two (two mergers)
    assert number_nodes_start - 2 == graph.vcount()


def test_merge_edges():
    """
    Test the merger of edge above a selected threshold
    """

    # create random graph
    graph = igraph.Graph.GRG(n=10, radius=2)

    name, clustering, cell = [], [], []

    # create attributes for the generated graph
    for i in graph.vs.indices:
        name.append(str(uuid.uuid4())[0:6] + str(i))
        clustering.append(str(uuid.uuid4())[0:6] + str(i))
        cell.append([str(uuid.uuid4())[0:6] + str(i), str(uuid.uuid4())[0:6] + str(i), str(uuid.uuid4())[0:6] + str(i)])

    edge_weights = np.random.rand(1, graph.ecount()).tolist()[0]
    threshold = max(edge_weights) * 0.9

    graph.es["weight"] = edge_weights

    graph.vs["name"] = name
    graph.vs["clustering"] = clustering
    graph.vs["cell"] = cell

    merged_graph = mcc.merge_edges_weight_above_threshold(graph, threshold)

    assert max(merged_graph.es["weight"]) < threshold


def test_merge_edge_meta_graph():
    """
    Test if edge weights and vertices are merged correctly.
    """
    # create the graph
    number_of_clusters_data = mcc.sort_by_number_clusters(settings_data, clustering_data, 4)
    graph = mcc.build_graph(number_of_clusters_data, clustering_data)

    # set the threshold
    threshold = 0.4

    # merge edges above threshold
    merged_graph = mcc.merge_edges_weight_above_threshold(graph, threshold)

    list_vertex_names = graph.vs["name"]

    list_edge_to_merge = []

    # select all edges that would be merged
    for edge in graph.es:
        if edge["weight"] >= threshold:
            list_edge_to_merge.append({edge.source, edge.target})

    # use the set intersection to group them
    list_vertices_to_merge = [list(vertex_group) for vertex_group in mcc.merge_list_of_sets(list_edge_to_merge)]

    # take the vertex indices and replace them with names
    merged_names = []
    for vertex_group in list_vertices_to_merge:
        vertex_group_names = []
        for vertex in vertex_group:
            vertex_group_names.append(graph.vs[vertex]["name"])

        merged_names.append(vertex_group_names)

    # test all edges above the threshold are merged
    assert max(merged_graph.es["weight"]) < threshold

    # assert vertices are merged correctly
    assert set(merged_names[0]) == set(merged_graph.vs["name"][0])

    assert 1 == len(merged_graph.vs["name"][1])

    assert set(merged_names[1]) == set(merged_graph.vs["name"][2])


def test_benchmark_speed():
    """
    Test the speed of the graph construction for multiple interations
    """

    # load parameters
    number_of_clusters_data = mcc.sort_by_number_clusters(settings_data, clustering_data, 4)
    number_of_iterations = 1000

    # measure time
    start = time.time()

    for i in range(number_of_iterations):
        mcc.build_graph(number_of_clusters_data, clustering_data)

    print((time.time() - start) / number_of_iterations)

