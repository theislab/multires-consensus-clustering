import multires_consensus_clustering as mcc
import pandas as pd
import numpy as np
import sklearn
import itertools
import matplotlib.pyplot as plt


def cluster_consistency(adata, clustering_data, settings_data, labels_df):
    """
    Check how much the labels differentiate throughout the runs.

    @param labels_df: True labels dataframe from the data directory
    @param settings_data: The settings data of the clusterings, number of clusters parameters, etc.
    @param clustering_data: The clustering data based of the adata file, as a pandas dataframe, e.g. cell, C001, ...
    @param adata: The single cell adata file.
    @return: The mean of the pairwise accuracies.
    """

    # set values for the mcc function
    connect_graph_neighbour_based = False
    multi_resolution_parameters = [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500]
    community_detection_function = "leiden"

    # list of all function parameters
    #function_parameters = [connect_graph_neighbour_based, community_detection_function, multi_resolution_parameters]

    # combination of all function parameters
    #combinations_parameters = list(itertools.product(*function_parameters))

    # print the number of combinations
    #print("Number of combinations: ", len(combinations_parameters))

    # create a dataframe for the cluster labels
    consistency_df = pd.DataFrame()

    # create a list of true labels based on the data given (NeurIPS)
    label_dict = dict(zip(labels_df["cell"], labels_df["cell_type"]))
    true_labels = [label_dict[cell] for cell in list(adata.obs.index)]

    # run evaluation for the multi resolution parameter
    for multi_resolution in multi_resolution_parameters:

        # set function name
        function_name = "MR_" + str(multi_resolution).replace('.', '')

        # multi resolution meta graph
        multires_graph = mcc.multiresolution_graph(clustering_data, settings_data, "all",
                                                   neighbour_based=connect_graph_neighbour_based)

        # multi resolution graph community detection
        multires_graph = mcc.multires_community_detection(multires_graph,
                                                          community_detection=community_detection_function,
                                                          clustering_data=clustering_data,
                                                          multi_resolution=multi_resolution)

        # create the final cluster labels
        df_clusters = mcc.graph_to_cell_labels_df(multires_graph)
        cluster_labels = mcc.df_cell_clusters_to_labels(df_clusters, adata, False)

        # add labels to dataframe
        consistency_df[function_name] = cluster_labels

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
