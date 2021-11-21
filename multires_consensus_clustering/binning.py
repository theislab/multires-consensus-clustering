def bin_n_clusters(clustering_sizes, min_clusterings=10):
    """
    Bin number of clusters

    Given the number of clusters for a set of clusterings divides them into
    bins where each bin contains at least ``min_clusterings`` clusterings.

    Parameters
    ----------
    clustering_sizes : pandas series with integer values
        Number of clusters in each clustering
    min_clusterings : int
        Minimum number of clusterings for each bin

    Returns
    -------
    list
        List where each item is a list containing the number of clusters groups
        for each bin
    """

    n_clusters_count = dict(clustering_sizes.sort_values().value_counts())

    bins = []
    current_bin = []
    current_clusterings = 0
    for n_clusters in clustering_sizes.sort_values().unique():
        n_clusterings = n_clusters_count[n_clusters]
        current_bin.append(n_clusters)
        current_clusterings += n_clusterings

        if current_clusterings >= min_clusterings:
            bins.append(current_bin)
            current_bin = []
            current_clusterings = 0

    if bins == []:
        bins = [current_bin]
    elif len(current_bin) > 0:
        bins[-1] = bins[-1] + current_bin

    return bins

