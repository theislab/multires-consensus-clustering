import multires_consensus_clustering as mcc
import pandas as pd
import numpy as np
import sklearn
import itertools
import matplotlib.pyplot as plt


def cluster_consistency(adata, clustering_data, settings_data):
    """
    Check how much the labels differentiate throughout the runs.

    @param settings_data: The settings data of the clusterings, number of clusters parameters, etc.
    @param clustering_data: The clustering data based of the adata file, as a pandas dataframe, e.g. cell, C001, ...
    @param adata: The single cell adata file.
    @return: The mean of the pairwise accuracies.
    """

    # set values for the mcc function
    connect_graph_neighbour_based = [True, False]
    merge_edges_threshold = [0.8, 0.9, 1.0]
    outlier_threshold = [0.5, 0.8, 0.9]
    community_detection_function = ["leiden", "component"]
    outlier_detection = ["probability"]

    # define number of runs
    number_runs = 5

    # list of all function parameters
    function_parameters = [connect_graph_neighbour_based, merge_edges_threshold, outlier_threshold,
                           community_detection_function]

    # combination of all function parameters
    combinations_parameters = list(itertools.product(*function_parameters))

    # print the number of combinations
    print("Number of combinations: ", len(combinations_parameters))

    # create a dataframe for the cluster labels
    consistency_df = pd.DataFrame()

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

        for index_test in range(number_runs):
            # multi resolution meta graph
            multires_graph = mcc.multiresolution_graph(clustering_data, settings_data, "all",
                                                       neighbour_based=connect_graph_neighbour_based)

            # multi resolution graph community detection
            multires_graph = mcc.multires_community_detection(multires_graph,
                                                              community_detection=community_detection_function,
                                                              merge_edges_threshold=merge_edges_threshold,
                                                              outlier_detection=outlier_detection,
                                                              outlier_detection_threshold=outlier_threshold,
                                                              clustering_data=clustering_data)

            # create the final cluster labels
            df_clusters = mcc.graph_to_cell_labels_df(multires_graph)
            cluster_labels = mcc.df_cell_clusters_to_labels(df_clusters, adata, False)

            consistency_df[function_name + "_" + str(index_test)] = cluster_labels

    # split dataframe columns back into the runs on the same function
    list_function_run = np.array_split(consistency_df.columns, len(consistency_df.columns) / number_runs)

    # create list for the scores
    scores_ami_all_runs, scores_rand_all_runs, list_function_used = [], [], []

    for run in list_function_run:

        # get all combinations of the created labels in the multiple runs
        combinations_of_clusters = list(itertools.combinations(run, 2))

        # create a list for the accuracy scores
        list_ami_scores, list_rand_scores = [], []

        # set name of the functions in the same run
        list_function_used.append(run[0][:-2])

        # calculate all accuracy scores and add them to the list
        for combination_labels in combinations_of_clusters:
            ami_score = sklearn.metrics.adjusted_mutual_info_score(consistency_df[combination_labels[0]],
                                                                   consistency_df[combination_labels[1]])

            #rand_score = sklearn.metrics.adjusted_rand_score(consistency_df[combination_labels[0]],
            #                                                consistency_df[combination_labels[1]])

            # add scores to list
            list_ami_scores.append(ami_score)
            #list_rand_scores.append(rand_score)

        scores_ami_all_runs.append(list_ami_scores)
        #scores_rand_all_runs.append(list_rand_scores)

    # create a df for the scores and plot these in a bar chart
    df_ami = create_df_scores_and_plot(scores_ami_all_runs, list_function_used, number_runs)
    #df_rand = create_df_scores_and_plot(scores_rand_all_runs, list_function_used, number_runs)

    return df_ami


def create_df_scores_and_plot(score_list, functions_used, number_runs):
    """
    Create a pandas df with the scores and the functions used and plot the results in a bar chart.
    @param score_list: A list of list wit the scores [[score_run_1, score_run_2, ...], [...],...]
    @param functions_used: A list of the names of the mcc functions used to create the scores.
    @param number_runs: The number the functions are run
    @return:
    """
    # create legend parameters
    combinations_of_runs = list(itertools.combinations(range(number_runs), 2))

    # create dataframe with the constancy cores
    df_consistency_scores = pd.DataFrame(score_list)
    df_consistency_scores.columns = ["AMI" + str(i) for i in combinations_of_runs]
    df_consistency_scores["mcc_functions"] = functions_used
    df_consistency_scores.set_index("mcc_functions", inplace=True)

    # create a bar chart for the scores
    df_consistency_scores.plot.bar()
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    return df_consistency_scores
