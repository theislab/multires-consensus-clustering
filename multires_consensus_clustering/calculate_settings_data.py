import pandas as pd
import numpy as np


def get_settings_data(clusterings_data):
    """
    Uses the clustering dataframe containg all clusterings for the cell data to create a dataframe containing the
        name of the cluserings and the number of clusters each of them contain.

    @param clusterings_data: Pandas Dataframe containg the cell names, the clusterings and the cluster labels for each cell
        e.g. [cells: [cell_name,...], C001: [1,2,3,...], C002: [1,2,1,...], ...]
    @return: Returns a dataframe with id of the clusterings and the number of clusters for each of them.
        E.g. [id: [C001, C002, ...], n_clusters: [5, 9, ...]
    """

    # create dataframe for the settings data
    settings_data = pd.DataFrame()

    # create lists for the clustering ids and n_clusters
    id_clusterings, list_n_clusters = [], []

    for column_name in clusterings_data.columns[1:]:
        # get all unique cluster labels from the current column of the clustering dataframe
        unique_cluster_labels = np.unique(clusterings_data[column_name].values)

        # get the number of clusters for the current clustering
        number_clusters = len(unique_cluster_labels)

        # add info from data frame to lists
        id_clusterings.append(column_name)
        list_n_clusters.append(number_clusters)

    # create a dataframe from the lists
    settings_data["id"] = id_clusterings
    settings_data["n_clusters"] = list_n_clusters

    return settings_data
