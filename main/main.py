from multires_consensus_clustering import Meta_Graph as mg
import multires_consensus_clustering as mcc
from pathlib import Path
import time

HERE = Path(__file__).parent.parent


start = time.time()

def meta_graph():
    """
    Uses the Meta Graph script to build the graph from the sc data.

    @return:
    """
    # read data
    clustering_data = mg.read_data(HERE / "data\s2d1_clustering.tsv", "all")
    settings_data = mg.read_data(HERE / "data\s2d1_settings.tsv", "all")


    # build graph, G is used as the variable for the Graph internally
    number_of_clusters_data = mg.sort_by_number_clusters(settings_data, clustering_data, 20)
    graph = mg.build_graph(number_of_clusters_data, clustering_data)

    # analyse grap
    # graph = merg_edges_weight_1(graph)
    #mg.plot_graph(graph, "label_off", "degree")

    #graph = min_cuts(graph)
    #graph_contracted = mcc.graph_community_detection(graph)

    #mcc.plot_interactive_graph(graph_contracted)


# run program
meta_graph()

# measure the time
end = time.time()
print("Time to run: ", end - start)