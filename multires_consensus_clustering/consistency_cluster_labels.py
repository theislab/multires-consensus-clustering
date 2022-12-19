import multires_consensus_clustering as mcc
import pandas as pd
import numpy as np
import sklearn
import itertools
import matplotlib.pyplot as plt


def cluster_consistency(adata, clustering_data, labels_df,
                        connect_graph_neighbour_based, community_detection_function,
                        single_community_detection_parameters, multi_community_detection_parameters,
                        bin_res, random_bin):
    """
    Check how much the labels differentiate throughout the runs. Each diffrent version of the method is compared to
        the true labels (labels_df) and evaluated.

    @param labels_df: True labels dataframe from the data directory
    @param clustering_data: The clustering data based of the adata file, as a pandas dataframe, e.g. cell, C001, ...
    @param adata: The single cell adata file.

    @param connect_graph_neighbour_based: Connects the graph either neighbor-based or connect all. E.g. [False]; [True]
    @param community_detection_function: The graph community detection function for the method. E.g. ["leiden"]

    @param single_community_detection_parameters: The resolution parameter for the graph community detection in the
        meta-graph step of the method. E.g. [0.1, 0.2, ..., 2]
    @param multi_community_detection_parameters: The resolution parameter for the community detection in the
        multi-resolution step. E.g. [0.1, 0.2, ..., 2]
    @param bin_res: The bin that should be selected for the evaluation. E.g. [0, 50, 100, 200, 300, 395], this will
        evaluate [(0,50), (0,100), (0,200), ..., (300,395)].
    @param random_bin: True or False. Either the clusterings are selected by random for each bin, or they are sorted.
        Meaning if False (0,50) has a smaller number of clusters than (300,395).

    @return: The scores of the evaluation as a pandas dataframe
    """

    # create a list of true labels based on the data given (NeurIPS)
    label_dict = dict(zip(labels_df["cell"], labels_df["cell_type"]))
    true_labels = [label_dict[cell] for cell in list(adata.obs.index)]

    # sort clustering_data by number of clusters
    clustering_data = sort_clustering_df_by_n_clusters(clustering_data)

    # create a list with all parameters
    graph_parameters = [connect_graph_neighbour_based, community_detection_function,
                        single_community_detection_parameters, multi_community_detection_parameters]

    # generate all possible combinations for the graph parameters
    graph_parameters = list(itertools.product(*graph_parameters))

    # print number of combinations for the graph parameters
    print("Number of graph parameters combinations", len(graph_parameters))

    # generate the resolutions for the bin selection
    bin_resolution = [(x, y) for x in bin_res for y in bin_res if x < y]

    # print number of combinations for bin resolutions
    print("Number of bin resolutions", len(bin_resolution))

    # create a dataframe for the cluster labels
    consistency_df = pd.DataFrame()

    # iterate thorough all bins
    for bin in bin_resolution:

        list_columns, min_n_clusters, max_n_clusters = select_columns(clustering_data, bin, random_clusters=random_bin)

        # run evaluation for the multi resolutions parameter
        for parameters in graph_parameters:

            # cut the clustering data based current parameters
            clustering_data_new = clustering_data[list_columns]

            # run the MRCC method and add the results to the dataframe
            consistency_df = creat_graph_to_df(clustering_data_new, adata, consistency_df, parameters,
                                               min_n_clusters, max_n_clusters)

    # run the evaluation for the generated labels
    evaluation_scores_df = evaluate_labels(true_labels, consistency_df)

    return evaluation_scores_df


def select_columns(clustering_data, bin, random_clusters):
    """
    Creates a list of column names based on the bin given as a parameter. This is used to only partially select
        clusterings from the clustering_data dataframe.

    @param clustering_data: The clustering data based of the adata file, as a pandas dataframe, e.g. cell, C001, ...
    @param bin: Tuple containing the number of clusterings to select. E.g (0,50) or (0,100).
    @param random_clusters: True or False, based on the parameter the bins are selected by random
        or sorted by number of clusters.
    @return: A list of labels [C001, C002, ...] selected based on the bin and random_clusters parameter,
        additionally returns the min and max number of cluster from all selected clusterings.
    """

    # select clusterings by random
    if random_clusters:
        # generate random indices for the column names list
        random_columns = np.random.random_integers(1, len(clustering_data.columns) - 1, bin[1])

        # remove duplicates
        random_columns = list(set(random_columns))

        # sort selected clustering indices
        random_columns.sort()

        # select the clustering-labels from the dataframe with the indices generated
        list_columns = list(clustering_data.columns[random_columns])

        # get the minimum and maximum number of cluster from the selected part of the dataframe
        min_n_clusters = np.min([np.min(clustering_data[column_name].values) for column_name in list_columns])
        max_n_clusters = np.max([np.max(clustering_data[column_name].values) for column_name in list_columns])

        # add cells to list of columns to keep the cell names
        list_columns = ["cell"] + list_columns

    # other wise select the clusterings sorted to only get a specific number of clusters.
    else:
        # check if the first bin is zero (the first clustering in the dataframe
        if bin[0] == 0:
            # get list of selected columns
            list_columns = list(clustering_data.columns[bin[0]:bin[1]])

            # get number of clusters for max and min parameters selected
            min_n_clusters = np.max(clustering_data.iloc[:, 1])
            max_n_clusters = np.max(clustering_data.iloc[:, bin[1]])

        # else starts with the
        else:
            list_columns = ["cell"] + list(clustering_data.columns[bin[0]:bin[1]])

            # get number of clusters for max and min parameters selected
            min_n_clusters = np.max(clustering_data.iloc[:, bin[0]])
            max_n_clusters = np.max(clustering_data.iloc[:, bin[1]])

    # return the calculated parameters
    return list_columns, min_n_clusters, max_n_clusters


def creat_graph_to_df(clustering_data, adata, results_df, graph_parameters, min_n_clusters, max_n_clusters):
    """
    Create the graph based on the mcc function and converts the graph to labels stored in a dataframe.

    @param clustering_data: The clusterings on which the graph is based (C001, C002, ...)
    @param adata: The adata file on which the single cell clustering is based.
    @param graph_parameters: The resolution, in which all parameters for the MRCC function are stored as a list.
        E.g. [[connect_graph_neighbour_based, community_detection_function, single_resolution, multi_resolution], ... ]
             = [[False, "leiden", 0.1, 1], [False, "leiden", 0.2, 1], ...]
    @param min_n_clusters: The lowest numbers of clusters used for the graph.
    @param max_n_clusters: The highest numbers of clusters used for the graph.
    @param results_df: The dataframe in which the results are saved.
    @return: The labels_df.
    """

    # calculate the settings data from the clustering data
    settings_data = mcc.get_settings_data(clustering_data)

    # set parameters for the MRCC function
    connect_graph_neighbour_based = graph_parameters[0]
    community_detection_function = graph_parameters[1]
    single_resolution = graph_parameters[2]
    multi_resolution = graph_parameters[3]

    # set function name
    function_name = "MRCC_n_clusters_" + str(min_n_clusters) + "_to_" + str(max_n_clusters) \
                    + "_total_" + str(len(clustering_data.columns)) \
                    + "_SR" + str(single_resolution) + "_MR" + str(multi_resolution)

    # multi resolution meta graph
    multires_graph = mcc.multiresolution_graph(clustering_data, settings_data, "all",
                                               neighbour_based=connect_graph_neighbour_based,
                                               single_resolution=single_resolution)

    # multi resolution graph community detection
    multires_graph = mcc.multires_community_detection(multires_graph,
                                                      community_detection=community_detection_function,
                                                      clustering_data=clustering_data,
                                                      multi_resolution=multi_resolution)

    # create the final cluster labels
    df_clusters = mcc.graph_to_cell_labels_df(multires_graph)
    cluster_labels = mcc.df_cell_clusters_to_labels(df_clusters, adata, False)

    # add labels to dataframe
    results_df[function_name] = cluster_labels

    return results_df


def evaluate_labels(true_labels, consistency_df):
    """
     Compares the true labels of the data with the generated labels from the MRCC function (stored in consistency_df).
        The evaluation scores are stored in the dataframe for alter plotting.

    @param true_labels: The true labels from the data.
    @param consistency_df: The labels generated by the MRCC function as dataframe with column names = function names and
        labels for rows, where each row index is the cell name.
    @return: A dataframe containing the function names as indices for the rows and scores in the first column.
    """

    # create list for the scores
    list_ami_scores, scores_rand_all_runs, list_function_used = [], [], []

    for column in consistency_df.columns:
        # get ami score of clustering labels and true labels
        ami_score = sklearn.metrics.adjusted_mutual_info_score(true_labels, consistency_df[column].values)

        # add scores to list
        list_ami_scores.append(ami_score)

    # create a df for the scores and plot these in a bar chart
    df_ami = create_df_scores_and_plot(consistency_df.columns, list_ami_scores)

    return df_ami


def create_df_scores_and_plot(functions_used, score_list):
    """
    Create a pandas df with the scores and the functions used and plot the results in a bar chart.

    @param score_list: A list of list wit the scores [[score_run_1, score_run_2, ...], [...],...]
    @param functions_used: A list of the names of the mcc functions used to create the scores.
    @return: Pandas dataframe with the scores as entries and function names as indices
    """

    # create dataframe with the constancy cores
    df_consistency_scores = pd.DataFrame()
    df_consistency_scores["mcc_functions"] = functions_used
    df_consistency_scores["mcc_scores"] = score_list

    # create a bar chart for the scores
    df_consistency_scores.plot.scatter(x="mcc_functions", y="mcc_scores")
    plt.xticks(rotation=90, ha='left')
    plt.tight_layout()
    plt.show()

    return df_consistency_scores


def sort_clustering_df_by_n_clusters(clustering_data):
    """
    Sorts the given clustering data by number of clusters.
        The new dataframe starts with cells, clusterings smallest number of clusters to highest.

    @param clustering_data: The clustering data based of the adata file, as a pandas dataframe, e.g. cell, C001, ...
    @return: The sorted dataframe.
    """

    # create new df for the transformed clustering labels (str -> int)
    sorting_df = pd.DataFrame()

    # get n_clusters for each clustering
    for column in clustering_data.columns[1:]:
        sorting_df[column] = [np.max(pd.to_numeric(clustering_data[column]))]

    # sort columns by n_clusters
    sorting_df = sorting_df.sort_values(by=0, ascending=True, axis=1)
    sorted_column_names = ["cell"] + list(sorting_df.columns)

    # rearrange the new dataframe based on the sorted columns
    clustering_data = clustering_data[sorted_column_names]

    return clustering_data

