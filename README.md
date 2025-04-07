Zero-shot SSL clustering
========================

This is the code to accompany the paper *An Empirical Study into Clustering of Unseen Datasets with Self-Supervised Encoders*, [arXiv:2406.02465](https://arxiv.org/abs/2406.02465).
For full details, please see [the paper](https://arxiv.org/abs/2406.02465).


Our recommendation for clustering embeddings
--------------------------------------------

As an output from our research, we propose the following method for clustering embeddings from a neural network.

```python
import sklearn.cluster
import umap


# 1. Create the embeddings
#
# For simplicity, we show one call of the encoder, but  if you have a lot of
# data to embed, you will need to call the encoder with batches of the data
# and then concatenate the result.
embeddings = encoder(data)

# 2. Reduce the data with UMAP
#
# Use between 5 and 100 components.
# Parameters n_neighbors=30 and min_dist=0.0 are set as recommended in the UMAP
# documentation when doing clustering.
# https://umap-learn.readthedocs.io/en/latest/clustering.html#umap-enhanced-clustering
reducer = umap.UMAP(
    n_neighbors=30,
    n_components=50,
    min_dist=0.0,
    metric="euclidean",
    random_state=rng_seed,  # Seed can be any int, set it for reproducibility
)
reduced_embds = reducer.fit_transform(embeddings)

# 3. Cluster the data
#
# Use Agglomerative Clustering on the UMAP-reduced embeddings.
# If you know how many clusters you are looking for, use that for n_clusters.
# Otherwise, you can either pick the distance_threshold which maximises the
# silhouette score, or the distance_threshold which maximises AMI on another
# dataset that is labelled.
clusterer = sklearn.cluster.AgglomerativeClustering(
    n_clusters=n_clusters,
    metric="euclidean",
    linkage="ward",
    distance_threshold=None,
)
clusterer.fit(reduced_embds)

# 4. Work with the cluster labels as you wish
y_pred = clusterer.labels_
# For example, you can compute AMI against ground-truth labels if known
ami = sklearn.metrics.adjusted_mutual_info_score(y_true, y_pred)
# or you can compute the intrinsic silhouette score in the original embedding
# space, or in the reduced embedding space.
ss_raw = sklearn.metrics.silhouette_score(embeddings, y_pred, metric="euclidean")
ss_red = sklearn.metrics.silhouette_score(reduced_embds, y_pred, metric="euclidean")
```


Citation
--------

If you find this work insightful, please consider citing [our paper](https://arxiv.org/abs/2406.02465).

```bibtex
@article{zero-shot-clustering,
    title={An Empirical Study into Clustering of Unseen Datasets with Self-Supervised Encoders},
    author={Scott C. Lowe and Joakim Bruslund Haurum and Sageev Oore and Thomas B. Moeslund and Graham W. Taylor},
    year={2024},
    eprint={2406.02465},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    journal={arXiv preprint},
    doi={10.48550/arxiv.2406.02465},
}
```
