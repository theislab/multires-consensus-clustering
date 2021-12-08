import scanpy as sc
import pandas as pd
import numpy as np
import constclust as cc


# Download data from https://openproblems.bio/neurips_docs/data/dataset/
# This script assumes it is being run in the downloaded "explore" directory

print("Loading the data...")
adata = sc.read_h5ad("cite/cite_gex_processed_training.h5ad")

print("Subsetting the s2d1 sample...")
adata_s2d1 = adata[adata.obs.batch == "s2d1", :].copy()
adata_s2d1

print("Calculating highly variable genes...")
sc.pp.highly_variable_genes(adata_s2d1, flavor="cell_ranger", n_top_genes=2000)

print("Calculating PCA...")
sc.tl.pca(adata_s2d1)

# Set up parameters for constclust
neighbours = [5, 15, 30, 50]
resolutions = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0]
states = [0, 1, 2]
distances = ["euclidean", "correlation", "cosine"]

print("Clustering dataset with constclust...")
results = {}
for distance in distances:
    print(f"Clustering using {distance} distance...")
    settings, clusterings = cc.cluster(
        adata_s2d1,
        n_neighbors=neighbours,
        resolutions=resolutions,
        random_state=states,
        neighbor_kwargs={"use_rep" : "X_pca", "metric" : distance}, 
        n_procs=4
    )
    settings["distance"] = distance
    settings["n_clusters"] = [clusterings[clustering].nunique() for clustering in clusterings.columns]
    results[distance] = {"settings" : settings, "clusterings" : clusterings}

print("Tidying results...")
settings = pd.concat([results[distance]["settings"] for distance in distances])
settings["id"] = [f"C{row:03}" for row in range(1, len(settings) + 1)]
settings = settings[["id", "distance", "n_neighbors", "resolution", "random_state", "n_clusters"]]

clusterings = pd.concat([results[distance]["clusterings"] for distance in distances], axis=1)
clusterings = clusterings.set_axis(settings["id"], axis=1)

print("Saving results...")
settings.to_csv("s2d1_settings.tsv", sep="\t", index=False)
clusterings.to_csv("s2d1_clustering.tsv", sep="\t", index_label="cell")

print("Done!")
