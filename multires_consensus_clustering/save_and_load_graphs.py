import string
from pathlib import Path
import igraph as ig

HERE = Path(__file__).parent.parent


def write_graph_to_file(graph, file_name):
    """
    Writes the graph to an python pickle file.
    @param neighbour_based: True or False, converted to string to save the graph of the neighbour based graph
        or completely connected graph.
    @param file_name: File name for the pickled graph.
    @param graph: The graph to be save.
    @return: The file of the saved graph.
    """

    # save graph to file
    file_path = "optimization_using_saved_graphs\saved_graphs\graph_neighbour_based_" + str(file_name) + ".pickle"
    file = graph.write_pickle(fname=HERE / file_path)

    return file


def load_graph_from_file(file_name):
    """
    Loads the python pickle file and converts it back to an iGraph graph.
    @param neighbour_based: True or False, converted to string to load the graph of the neighbour based graph
        or completely connected graph.
    @param file_name: File name of the pickled graph.
    @return: Returns the graph created from the pickle file.
    """

    # read graph from file
    file_path = "optimization_using_saved_graphs\saved_graphs" + chr(92) + file_name
    graph = ig.Graph.Read_Pickle(fname=HERE / file_path)

    return graph
