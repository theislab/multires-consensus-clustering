from pathlib import Path
import igraph as ig

HERE = Path(__file__).parent.parent


def write_graph_to_file(graph, neighbour_based):
    """
    Writes the graph to an python pickle file.
    @param neighbour_based: True or False, converted to string to save the graph of the neighbour based graph
        or completely connected graph.
    @param graph: The graph to be save.
    @return: The file of teh saved graph.
    """

    # save graph to file
    file_path = "optimization_using_saved_graphs\saved_graphs\graph_neighbour_based_" + str(neighbour_based) + ".pickle"
    file = graph.write_pickle(fname=HERE / file_path)

    return file


def load_graph_from_file(neighbour_based):
    """
    Loads the pytzhon pcikel file and converts it back to an iGraph graph.
    @param neighbour_based: True or False, converted to string to load the graph of the neighbour based graph
        or completely connected graph.
    @return: Returns the graph created from the pickle file.
    """

    # read graph from file
    file_path = "optimization_using_saved_graphs\saved_graphs\graph_neighbour_based_" + str(neighbour_based) + ".pickle"
    graph = ig.Graph.Read_Pickle(fname=HERE / file_path)

    return graph
