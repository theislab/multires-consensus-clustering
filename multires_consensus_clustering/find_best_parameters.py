import multires_consensus_clustering as mcc
import pandas as pd
import numpy as np
import sklearn
import itertools
import matplotlib.pyplot as plt


def find_best_parameters(adata, clustering_data, settings_data, labels_df):
    """
    Check how much the labels differentiate throughout the runs.

    @param labels_df: A pandas dataframe containg the true labels for the clustering data.
    @param settings_data: The settings data of the clusterings, number of clusters parameters, etc.
    @param clustering_data: The clustering data based of the adata file, as a pandas dataframe, e.g. cell, C001, ...
    @param adata: The single cell adata file.
    @return: The mean of the pairwise accuracies.
    """

    # set values for the mcc function
    connect_graph_neighbour_based = [False]
    merge_edges_threshold = [0.9]
    outlier_threshold = [0.95]
    community_detection_function = ["component", "leiden", "louvain"]

    # list of all function parameters
    function_parameters = [connect_graph_neighbour_based, merge_edges_threshold, outlier_threshold,
                           community_detection_function]

    # combination of all function parameters
    combinations_parameters = list(itertools.product(*function_parameters))

    # print the number of combinations
    print("Number of combinations: ", len(combinations_parameters))

    # create list for functions names
    list_function_names = []

    # create lists for the scores
    ami_list, rand_list = [], []

    # load true labels
    label_dict = dict(zip(labels_df["cell"], labels_df["cell_type"]))
    new_labels = [label_dict[cell] for cell in list(adata.obs.index)]

    # run the mcc function multiple times
    for parameters in combinations_parameters:

        # set parameters
        connect_graph_neighbour_based = parameters[0]
        merge_edges_threshold = parameters[1]
        outlier_threshold = parameters[2]
        community_detection_function = parameters[3]

        # set function name
        if connect_graph_neighbour_based:
            connect_graph = "N"
        else:
            connect_graph = "A"

        if community_detection_function == "leiden":
            comm_detect = "Le"
        elif community_detection_function == "component":
            comm_detect = "Co"
        else:
            comm_detect = "Lo"

        function_name = "MRCC_" + connect_graph + "_" + comm_detect + "_O" + str(outlier_threshold).replace('.', '') + \
                        "_M" + str(merge_edges_threshold).replace('.', '')

        multires_graph = mcc.multiresolution_graph(clustering_data, settings_data, "all",
                                                   neighbour_based=connect_graph_neighbour_based)

        # multi resolution graph community detection
        multires_graph = mcc.multires_community_detection(multires_graph,
                                                          community_detection=community_detection_function,
                                                          merge_edges_threshold=merge_edges_threshold,
                                                          outlier_detection="probability",
                                                          outlier_detection_threshold=outlier_threshold,
                                                          clustering_data=clustering_data)

        # create the final cluster labels
        df_clusters = mcc.graph_to_cell_labels_df(multires_graph)
        cluster_labels = mcc.df_cell_clusters_to_labels(df_clusters, adata, False)

        # set name of the functions in the same run
        list_function_names.append(function_name)

        ami_score = sklearn.metrics.adjusted_mutual_info_score(cluster_labels, new_labels)

        # rand_score = sklearn.metrics.adjusted_rand_score(cluster_labels, new_labels)

        ami_list.append(ami_score)
        # rand_list.append(rand_score)

    # create a df for the scores and plot these in a bar chart
    df_ami = create_df_scores_and_plot(ami_list, list_function_names)
    # df_rand = create_df_scores_and_plot(rand_list, list_function_names)

    return df_ami


def create_df_scores_and_plot(score_list, functions_used):
    """
    Create a pandas df with the scores and the functions used and plot the results in a bar chart.
    @param score_list: A list of list wit the scores [[score_run_1, score_run_2, ...], [...],...]
    @param functions_used: A list of the names of the mcc functions used to create the scores.
    @return:
    """

    # create dataframe with the constancy cores
    df_consistency_scores = pd.DataFrame(score_list)
    df_consistency_scores.columns = ["AMI"]
    df_consistency_scores["mcc_functions"] = functions_used
    df_consistency_scores.set_index("mcc_functions", inplace=True)

    # print dataframe
    print(df_consistency_scores)

    # create a bar chart for the scores
    df_consistency_scores.plot.bar()
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    return df_consistency_scores
