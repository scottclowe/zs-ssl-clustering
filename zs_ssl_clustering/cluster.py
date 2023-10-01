#!/usr/bin/env python

import copy
import os
import random
import time
from datetime import datetime

import numpy as np
import psutil
import sklearn.cluster
import sklearn.manifold
import sklearn.metrics
from sklearn.decomposition import PCA, KernelPCA

from zs_ssl_clustering import io, utils

CLUSTERERS = [
    "KMeans",
    "AffinityPropagation",
    "AgglomerativeClustering",
    "SpectralClustering",
    "HDBSCAN",
    "OPTICS",
]

METRICS = [
    "arccos",  # Manually implemented as unit length norm + euclidean distance
    "braycurtis",  # Like L1, but weights the result
    "canberra",  # Like L1, but weights dimensions by their magnitude
    "chebyshev",  # L-infinity
    "cityblock",  # L1
    "cosine",  # Supported by AgglomerativeClustering and OPTICS
    "euclidean",  # L2
    "infinity",
    "l1",
    "l2",
    "mahalanobis",  # Must provide either V or VI in ``metric_params``.
    "manhattan",  # L1
    "minkowski",  # Lp norm, Must provide a p value in ``p`` or ``metric_params``.
    "p",
    "seuclidean",  # Weighted L2. Needs an argument ``V`` with variances per dim.
]


def run(config):
    print("Configuration:")
    print()
    print(config)
    print(flush=True)

    if config.seed is not None:
        # Seed RNG state for reproducibility
        random.seed(config.seed)
        np.random.seed(config.seed % 0xFFFF_FFFF)

    if config.zscore is None:
        # If z-score was not specified, default to True if PCA is used, False otherwise.
        config.zscore = "PCA" in config.dim_reducer

    if config.distance_metric == "arccos":
        # If arccos distance metric is used, we need to normalize the vectors
        # to unit length.
        config.normalize = True

    config.memory_total_GB = psutil.virtual_memory().total / 1_000_000_000

    # Handle arccos distance metric by normalizing the vectors ourself and
    # passing euclidean to the clusterer, since it doesn't support arccos directly.
    _distance_metric = config.distance_metric
    _distance_metric = "euclidean" if _distance_metric == "arccos" else _distance_metric

    _distance_metric_man = config.dim_reducer_man_metric
    _distance_metric_man = (
        "euclidean" if _distance_metric_man == "arccos" else _distance_metric_man
    )

    if config.log_wandb:
        # Lazy import of wandb, since logging to wandb is optional
        import wandb

        wandb_run_name = config.run_name
        if wandb_run_name is not None and config.run_id is not None:
            wandb_run_name = f"{wandb_run_name}__{config.run_id}"
        wandb.init(
            name=wandb_run_name,
            id=config.run_id,
            entity=config.wandb_entity,
            project=config.wandb_project,
            group=config.wandb_group,
            config=config,
            job_type="cluster",
            tags=config.wandb_tags,
            config_exclude_keys=[
                "log_wandb",
                "wandb_entity",
                "wandb_project",
                "wandb_tags",
                "wandb_group",
                "run_name",
                "run_id",
            ],
        )
        # If a run_id was not supplied at the command prompt, wandb will
        # generate a name. Let's use that as the run_id.
        if config.run_id is None:
            config.run_id = wandb.run.name

    # If we still don't have a run_id, generate one from the current time.
    if config.run_id is None:
        config.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Only need allow_pickle=True if we're using the saved config dict
    data = np.load(io.get_embeddings_path(config))
    embeddings = data["embeddings"]
    y_true = data["y_true"]
    n_clusters_gt = len(np.unique(y_true))
    encoding_dim = embeddings.shape[-1]

    og_embeddings = copy.deepcopy(embeddings)

    clusterer_args_used = set()
    results = {}

    start_reducing = time.time()
    if config.zscore:
        # Standardize to zero mean, unit variance
        embeddings -= np.mean(embeddings, axis=0)
        sigma = np.std(embeddings, axis=0)
        embeddings /= sigma
        # Handle the case where a dimension has zero variance
        embeddings[:, sigma == 0] = 0

    if config.dim_reducer is None or config.dim_reducer == "None":
        if config.log_wandb:
            wandb.config.update({"pca_kernel": None}, allow_val_change=True)
    elif "PCA" in config.dim_reducer:
        use_kernel_PCA = config.dim_reducer == "KernelPCA"
        if not use_kernel_PCA:
            config.pca_kernel = "none"
            if config.log_wandb:
                wandb.config.update({"pca_kernel": "none"}, allow_val_change=True)
        ndim_reduced = config.ndim_reduced
        pca_variance = config.pca_variance

        if ndim_reduced is None and pca_variance is None and not use_kernel_PCA:
            raise ValueError("Neither 'ndim_reduced' nor 'pca_variance' was specified")
        elif ndim_reduced is not None and pca_variance is not None:
            raise ValueError("Both 'ndim_reduced' and 'pca_variance' was specified")

        if pca_variance is not None:
            if use_kernel_PCA:
                raise ValueError(
                    "Cannot use KernelPCA by specifying the variance to be kept"
                )
            assert isinstance(pca_variance, float), "pca_variance must be float"
            assert (
                pca_variance > 0.0 and pca_variance < 1.0
            ), "pca_variance must be between 0 and 1"
            n_components = pca_variance
        elif ndim_reduced is None:
            if not use_kernel_PCA:
                raise ValueError(
                    "Neither 'ndim_reduced' nor 'pca_variance' was specified"
                )
            n_components = ndim_reduced
        elif ndim_reduced == "mle":
            if use_kernel_PCA:
                raise ValueError(
                    "Cannot use Minka's MLE is used to guess the dimension with KernelPCA."
                )
            n_components = ndim_reduced
        elif ndim_reduced.isdecimal():
            n_components = int(ndim_reduced)
        else:
            raise ValueError(
                f"Unrecognized value for 'ndim_reduced': '{ndim_reduced}'."
                " Should be 'mle' or an integer value."
            )

        start_pca = time.time()
        if use_kernel_PCA:
            pca = KernelPCA(
                n_components=n_components,
                kernel=config.pca_kernel,
                random_state=config.seed,
                n_jobs=config.workers,
            )
            clusterer_args_used = clusterer_args_used.union(
                {"seed", "workers", "pca_kernel"}
            )
        else:
            pca = PCA(n_components=n_components, random_state=config.seed)
            clusterer_args_used.add("seed")
        embeddings = pca.fit_transform(embeddings)
        end_pca = time.time()
        results["time_pca"] = end_pca - start_pca

        shape_after = embeddings.shape[-1]

        print(f"Shape Before/After PCA: {encoding_dim}/{shape_after}")
        if not use_kernel_PCA:
            print(
                f"PCA Explained Variance: {np.sum(pca.explained_variance_ratio_)*100} %"
            )
            results["pca_explained_ratio"] = np.sum(pca.explained_variance_ratio_)
        print(f"{config.dim_reducer} fitting time: {results['time_pca']:.2f}s")
    else:
        raise ValueError(
            f"Unrecognized dimensionality reduction method: '{config.dim_reducer}'"
        )

    if config.dim_reducer_man is None or config.dim_reducer_man == "None":
        if config.log_wandb:
            wandb.config.update({"dim_reducer_man_metric": None}, allow_val_change=True)
        reducerman_args_used = set()

    elif config.dim_reducer_man == "UMAP":
        if config.ndim_reduced_man is None:
            raise ValueError(
                f"{config.dim_reducer_man} reduction was requested, but 'ndim_reduced_man' was not set."
            )

        import umap

        _embeddings = embeddings
        if config.dim_reducer_man_metric == "arccos":
            _embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        start_reduce_man = time.time()
        reducer_man = umap.UMAP(
            n_neighbors=30,
            n_components=config.ndim_reduced_man,
            min_dist=0.0,
            metric=_distance_metric_man,
            random_state=config.seed,
            n_jobs=config.workers,  # Only 1 worker used if RNG is manually seeded
            verbose=config.verbose > 0,
        )
        reducerman_args_used = {"dim_reducer_man_metric", "seed"}
        embeddings = reducer_man.fit_transform(_embeddings)
        end_reduce_man = time.time()
        results["time_reduce_man"] = end_reduce_man - start_reduce_man
        print(
            f"{config.dim_reducer_man} fitting time: {results['time_reduce_man']:.2f}s"
        )

    elif config.dim_reducer_man == "tSNE":
        if config.ndim_reduced_man is None:
            raise ValueError(
                f"{config.dim_reducer_man} reduction was requested, but 'ndim_reduced_man' was not set."
            )

        _embeddings = embeddings
        if config.dim_reducer_man_metric == "arccos":
            _embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        start_reduce_man = time.time()
        reducer_man = sklearn.manifold.TSNE(
            n_components=config.ndim_reduced_man,
            metric=_distance_metric_man,
            verbose=config.verbose,
            random_state=config.seed,
            method="exact",  # The default, "barnes_hut", only supports n_components<4
            n_jobs=config.workers,
        )
        reducerman_args_used = {"dim_reducer_man_metric", "seed", "workers"}
        embeddings = reducer_man.fit_transform(_embeddings)
        end_reduce_man = time.time()
        results["time_reduce_man"] = end_reduce_man - start_reduce_man
        print(
            f"{config.dim_reducer_man} fitting time: {results['time_reduce_man']:.2f}s"
        )

    else:
        raise ValueError(
            f"Unrecognized nearest-neighbour dimensionality reduction method: '{config.dim_reducer_man}'"
        )

    end_reducing = time.time()

    reduced_dim = embeddings.shape[-1]

    clusterer_args = {
        "distance_metric",
        "max_iter",
        "min_samples",
        "max_samples",
        "workers",
        "affinity_damping",
        "affinity_conv_iter",
        "spectral_assigner",
        "aggclust_linkage",
        "aggclust_dist_thresh",
        "hdbscan_method",
        "optics_method",
        "optics_xi",
    }

    if config.distance_metric == "arccos":
        clusterer_args_used.add("distance_metric")

    if config.clusterer_name == "KMeans":
        clusterer = sklearn.cluster.KMeans(
            n_clusters=n_clusters_gt,
            random_state=config.seed,
            max_iter=config.max_iter,
            init="k-means++",
            n_init=1,
            verbose=config.verbose,
        )
        clusterer_args_used = clusterer_args_used.union({"seed", "max_iter"})

    elif config.clusterer_name == "AffinityPropagation":
        clusterer = sklearn.cluster.AffinityPropagation(
            damping=config.affinity_damping,
            max_iter=config.max_iter,
            convergence_iter=config.affinity_conv_iter,
            verbose=config.verbose > 0,
            random_state=config.seed,
        )
        clusterer_args_used = clusterer_args_used.union(
            {
                "seed",
                "max_iter",
                "affinity_damping",
                "affinity_conv_iter",
            }
        )

    elif config.clusterer_name == "SpectralClustering":
        # TODO Look into this:
        # Requires the number of clusters
        # Can be estimated through e.g.
        # https://proceedings.neurips.cc/paper_files/paper/2004/file/40173ea48d9567f1f393b20c855bb40b-Paper.pdf
        # Might be more recent work to consider
        clusterer = sklearn.cluster.SpectralClustering(
            n_clusters=n_clusters_gt,
            assign_labels=config.spectral_assigner,
            random_state=config.seed,
            verbose=config.verbose > 0,
        )
        clusterer_args_used = clusterer_args_used.union({"seed", "spectral_assigner"})

    elif config.clusterer_name == "AgglomerativeClustering":
        # Can work with specified number of clusters, as well as unknown (which requires a distance threshold)
        # We can also impose some structure metric through the "connectivity" argument
        if config.aggclust_dist_thresh is None:
            n_clusters = n_clusters_gt
        else:
            n_clusters = None
        clusterer = sklearn.cluster.AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=_distance_metric,
            linkage=config.aggclust_linkage,
            distance_threshold=config.aggclust_dist_thresh,
        )
        clusterer_args_used = clusterer_args_used.union(
            {
                "distance_metric",
                "aggclust_linkage",
                "aggclust_dist_thresh",
            }
        )

    elif config.clusterer_name == "HDBSCAN":
        max_cluster_size = config.max_samples
        if max_cluster_size is None:
            pass
        elif max_cluster_size < 1.0:
            max_cluster_size = round(max_cluster_size * len(embeddings))
        else:
            max_cluster_size = int(max_cluster_size)
        clusterer = sklearn.cluster.HDBSCAN(
            min_cluster_size=config.min_samples,
            max_cluster_size=max_cluster_size,
            metric=_distance_metric,
            cluster_selection_method=config.hdbscan_method,
            n_jobs=config.workers,
        )
        clusterer_args_used = clusterer_args_used.union(
            {
                "min_samples",
                "max_samples",
                "distance_metric",
                "hdbscan_method",
                "workers",
            }
        )

    elif config.clusterer_name == "OPTICS":
        clusterer = sklearn.cluster.OPTICS(
            min_samples=config.min_samples,
            metric=_distance_metric,
            cluster_method=config.optics_method,
            xi=config.optics_xi,
            n_jobs=config.workers,
        )
        clusterer_args_used = clusterer_args_used.union(
            {
                "min_samples",
                "distance_metric",
                "optics_method",
                "workers",
            }
        )
        if config.optics_method == "xi":
            clusterer_args_used.add("optics_xi")

    else:
        raise ValueError(f"Unrecognized clusterer: '{config.clusterer_name}'")

    # Standardize to zero mean, unit variance
    zs2_embeddings = embeddings - np.mean(embeddings, axis=0)
    sigma = np.std(zs2_embeddings, axis=0)
    zs2_embeddings /= sigma
    # Handle the case where a dimension has zero variance
    zs2_embeddings[:, sigma == 0] = 0
    # Correct for distances scaling up with the number of dimensions
    zs2_embeddings /= np.sqrt(zs2_embeddings.shape[-1])

    # Normalize embeddings to have unit length
    nrm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Normalize z-scored embeddings to have unit length
    nrm_zs2_embeddings = zs2_embeddings / np.linalg.norm(
        zs2_embeddings, axis=1, keepdims=True
    )

    _embeddings = embeddings

    if config.zscore2 and (config.normalize or config.distance_metric == "arccos"):
        # Fit clusterer on embeddings that are z-scored and then normalized
        _embeddings = nrm_zs2_embeddings

    elif config.zscore2:
        # Fit clusterer on z-scored embeddings
        _embeddings = zs2_embeddings

    elif config.normalize or config.distance_metric == "arccos":
        # Fit clusterer on normalized embeddings
        _embeddings = nrm_embeddings

    # Wipe the state of cluster arguments that were not relevant to the
    # chosen clusterer.
    clusterer_args_unused = clusterer_args.difference(reducerman_args_used)
    clusterer_args_unused = clusterer_args_unused.difference(clusterer_args_used)
    for key in clusterer_args_unused:
        setattr(config, key, None)
        if config.log_wandb:
            wandb.config.update({key: None}, allow_val_change=True)
    if config.log_wandb and config.workers is not None and config.workers == -1:
        wandb.config.update(
            {"workers": utils.get_num_cpu_available()}, allow_val_change=True
        )

    start_cluster = time.time()
    clusterer.fit(_embeddings)
    end_cluster = time.time()

    y_pred = clusterer.labels_
    select_clustered = y_pred >= 0
    n_clusters_pred = len(np.unique(y_pred[select_clustered]))
    ratio_clustered = np.sum(select_clustered) / len(y_pred)
    ratio_unclustered = 1 - ratio_clustered

    # Load raw (flattened only) images for computing intrinsic metrics
    raw_data = np.load(io.get_embeddings_path(config), raw=True)
    raw_images = raw_data["embeddings"]

    _results = {
        "n_samples": len(embeddings),
        "encoding_dim": encoding_dim,
        "reduced_dim": reduced_dim,
        "time_reducing": end_reducing - start_reducing,
        "time_clustering": end_cluster - start_cluster,
        "y_pred": y_pred,
        "num_cluster_true": n_clusters_gt,
        "num_cluster_pred": n_clusters_pred,
        "ratio_clustered": ratio_clustered,
        "ratio_unclustered": ratio_unclustered,
        "AMI": sklearn.metrics.adjusted_mutual_info_score(y_true, y_pred),
        "ARI": sklearn.metrics.adjusted_rand_score(y_true, y_pred),
        "FMS": sklearn.metrics.fowlkes_mallows_score(y_true, y_pred),
        "completeness": sklearn.metrics.completeness_score(y_true, y_pred),
        "homogeneity": sklearn.metrics.homogeneity_score(y_true, y_pred),
        "CHS_true": sklearn.metrics.calinski_harabasz_score(embeddings, y_true),
        "CHS-nrm_true": sklearn.metrics.calinski_harabasz_score(nrm_embeddings, y_true),
        "CHS-zs2_true": sklearn.metrics.calinski_harabasz_score(zs2_embeddings, y_true),
        "CHS-zs2-nrm_true": sklearn.metrics.calinski_harabasz_score(
            nrm_zs2_embeddings, y_true
        ),
        "CHS-og_true": sklearn.metrics.calinski_harabasz_score(og_embeddings, y_true),
        "CHS-pix_true": sklearn.metrics.calinski_harabasz_score(raw_images, y_true),
        "DBS_true": sklearn.metrics.davies_bouldin_score(embeddings, y_true),
        "DBS-nrm_true": sklearn.metrics.davies_bouldin_score(nrm_embeddings, y_true),
        "DBS-zs2_true": sklearn.metrics.davies_bouldin_score(zs2_embeddings, y_true),
        "DBS-zs2-nrm_true": sklearn.metrics.davies_bouldin_score(
            nrm_zs2_embeddings, y_true
        ),
        "DBS-og_true": sklearn.metrics.davies_bouldin_score(og_embeddings, y_true),
        "DBS-pix_true": sklearn.metrics.calinski_harabasz_score(raw_images, y_true),
    }
    results.update(_results)

    if n_clusters_pred > 1 and len(np.unique(y_pred)) < len(embeddings):
        results["CHS_pred"] = sklearn.metrics.calinski_harabasz_score(
            embeddings, y_pred
        )
        results["CHS-nrm_pred"] = sklearn.metrics.calinski_harabasz_score(
            nrm_embeddings, y_pred
        )
        results["CHS-zs2_pred"] = sklearn.metrics.calinski_harabasz_score(
            zs2_embeddings, y_pred
        )
        results["CHS-zs2-nrm_pred"] = sklearn.metrics.calinski_harabasz_score(
            nrm_zs2_embeddings, y_pred
        )
        results["CHS-og_pred"] = sklearn.metrics.calinski_harabasz_score(
            og_embeddings, y_pred
        )
        results["CHS-pix_pred"] = sklearn.metrics.calinski_harabasz_score(
            raw_images, y_pred
        )
        results["DBS_pred"] = sklearn.metrics.davies_bouldin_score(embeddings, y_pred)
        results["DBS-nrm_pred"] = sklearn.metrics.davies_bouldin_score(
            nrm_embeddings, y_pred
        )
        results["DBS-zs2_pred"] = sklearn.metrics.davies_bouldin_score(
            zs2_embeddings, y_pred
        )
        results["DBS-zs2-nrm_pred"] = sklearn.metrics.davies_bouldin_score(
            nrm_zs2_embeddings, y_pred
        )
        results["DBS-og_pred"] = sklearn.metrics.davies_bouldin_score(
            og_embeddings, y_pred
        )
        results["DBS-pix_pred"] = sklearn.metrics.davies_bouldin_score(
            raw_images, y_pred
        )

    # Repeat metrics, but considering only the samples that were clustered
    if ratio_clustered > 0:
        yct = y_true[select_clustered]
        ycp = y_pred[select_clustered]
        ec = embeddings[select_clustered]
        results["AMI_clus"] = sklearn.metrics.adjusted_mutual_info_score(yct, ycp)
        results["ARI_clus"] = sklearn.metrics.adjusted_rand_score(yct, ycp)
        results["FMS_clus"] = sklearn.metrics.fowlkes_mallows_score(yct, ycp)
        results["completeness_clus"] = sklearn.metrics.completeness_score(yct, ycp)
        results["homogeneity_clus"] = sklearn.metrics.homogeneity_score(yct, ycp)
        if n_clusters_pred > 1 and n_clusters_pred < len(ec):
            results["CHS_pred_clus"] = sklearn.metrics.calinski_harabasz_score(ec, ycp)
            results["CHS-nrm_pred_clus"] = sklearn.metrics.calinski_harabasz_score(
                nrm_embeddings[select_clustered], ycp
            )
            results["CHS-zs2_pred_clus"] = sklearn.metrics.calinski_harabasz_score(
                zs2_embeddings[select_clustered], ycp
            )
            results["CHS-zs2-nrm_pred_clus"] = sklearn.metrics.calinski_harabasz_score(
                nrm_zs2_embeddings[select_clustered], ycp
            )
            results["CHS-og_pred_clus"] = sklearn.metrics.calinski_harabasz_score(
                og_embeddings[select_clustered], ycp
            )
            results["CHS-pix_pred_clus"] = sklearn.metrics.calinski_harabasz_score(
                raw_images[select_clustered], ycp
            )
            results["DBS_pred_clus"] = sklearn.metrics.davies_bouldin_score(ec, ycp)
            results["DBS-nrm_pred_clus"] = sklearn.metrics.davies_bouldin_score(
                nrm_embeddings[select_clustered], ycp
            )
            results["DBS-zs2_pred_clus"] = sklearn.metrics.davies_bouldin_score(
                zs2_embeddings[select_clustered], ycp
            )
            results["DBS-zs2-nrm_pred_clus"] = sklearn.metrics.davies_bouldin_score(
                nrm_zs2_embeddings[select_clustered], ycp
            )
            results["DBS-og_pred_clus"] = sklearn.metrics.davies_bouldin_score(
                og_embeddings[select_clustered], ycp
            )
            results["DBS-pix_pred_clus"] = sklearn.metrics.davies_bouldin_score(
                raw_images[select_clustered], ycp
            )

    # Compute silhouette scores with several distance metrics
    for dm in ["euclidean", "l1", "chebyshev", "arccos", "braycurtis", "canberra"]:
        for space_name, embs in [
            ("reduced", embeddings),
            ("nrm", nrm_embeddings),
            ("zs2", zs2_embeddings),
            ("zs2-nrm", nrm_zs2_embeddings),
            ("og", og_embeddings),
            ("pix", raw_images),
        ]:
            if space_name == "reduced":
                prefix = f"silhouette-{dm}"
            else:
                prefix = f"silhouette-{space_name}-{dm}"
            my_dstmtr = dm
            my_embs = embs
            if dm == "arccos":
                my_dstmtr = "euclidean"
                my_embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

            # Compute metrics on ground-truth clusters
            try:
                results[f"{prefix}_true"] = sklearn.metrics.silhouette_score(
                    my_embs, y_true, metric=my_dstmtr
                )
            except Exception as err:
                print(f"Error computing GT silhouette score with {dm}: {err}")

            # Compute metrics on ground-truth clusters, but considering only the
            # samples that were clustered
            if ratio_clustered > 0:
                try:
                    results[f"{prefix}_true_clus"] = sklearn.metrics.silhouette_score(
                        my_embs[select_clustered], yct, metric=my_dstmtr
                    )
                except Exception as err:
                    print(f"Error computing pred silhouette score with {dm}: {err}")

            # Compute metrics on predicted clusters
            if n_clusters_pred <= 1 or len(np.unique(y_pred)) == len(embeddings):
                continue
            try:
                results[f"{prefix}_pred"] = sklearn.metrics.silhouette_score(
                    my_embs, y_pred, metric=my_dstmtr
                )
            except Exception as err:
                print(
                    f"Error computing GT (clustered samples only) silhouette score with {dm}: {err}"
                )

            # Compute metrics on predicted clusters, but considering only the
            # samples that were clustered
            if ratio_clustered <= 0 or n_clusters_pred >= len(ec):
                continue
            try:
                results[f"{prefix}_pred_clus"] = sklearn.metrics.silhouette_score(
                    my_embs[select_clustered], ycp, metric=my_dstmtr
                )
            except Exception as err:
                print(
                    f"Error computing pred (clustered samples only) silhouette score with {dm}: {err}"
                )

    if hasattr(clusterer, "n_iter_"):
        results["iter"] = clusterer.n_iter_  # Number of iterations run.
        results["converged"] = clusterer.n_iter_ < config.max_iter

    print(
        f"\n{config.clusterer_name}({config.model}({config.dataset_name}))"
        " evaluation results:"
    )
    for k, v in results.items():
        if "time" in k:
            print(f"  {k + ' ':.<36s} {v:10.4f} seconds")
        elif isinstance(k, int):
            print(f"  {k + ' ':.<36s} {v:>5d}")
        else:
            try:
                print(f"  {k + ' ':.<36s} {v:10.4f}")
            except TypeError:
                print(f"  {k + ' ':.<36s} {v}")

    if config.log_wandb:
        wandb.log(results)

    return results


def get_parser():
    r"""
    Build argument parser for the command line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser.
    """
    import argparse
    import sys

    # Use the name of the file called to determine the name of the program
    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        # If the file is called __main__.py, go up a level to the module name
        prog = os.path.split(__file__)[1]
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Cluster extracted embeddings.",
        add_help=False,
    )
    # Help arg ----------------------------------------------------------------
    group = parser.add_argument_group("Help")
    group.add_argument(
        "--help",
        "-h",
        action="help",
        help="Show this help message and exit.",
    )
    # Dataset args ------------------------------------------------------------
    group = parser.add_argument_group("Dataset")
    group.add_argument(
        "--dataset",
        dest="dataset_name",
        type=str,
        default="cifar10",
        help="Name of the dataset to learn. Default: %(default)s",
    )
    group.add_argument(
        "--partition",
        type=str,
        default="test",
        help="Which partition of the dataset to use. Default: %(default)s",
    )
    group.add_argument(
        "--zoom-ratio",
        type=float,
        default=1.0,
        help="Ratio of how much of the image to zoom in on. Default: %(default)s",
    )
    # Architecture args -------------------------------------------------------
    group = parser.add_argument_group("Architecture")
    group.add_argument(
        "--model",
        "--encoder",
        dest="model",
        type=str,
        default="resnet18",
        help="Name of model architecture. Default: %(default)s",
    )
    # Output checkpoint args --------------------------------------------------
    group = parser.add_argument_group("Output checkpoint")
    group.add_argument(
        "--embedding-dir",
        type=str,
        metavar="PATH",
        default="embeddings",
        help="Path to directory containing embeddings.",
    )
    # Reproducibility args ----------------------------------------------------
    group = parser.add_argument_group("Reproducibility")
    group.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random number generator (RNG) seed. Default: %(default)s",
    )
    # Dimensionality reduction args -------------------------------------------
    group = parser.add_argument_group("Dimensionality reduction (variance/PCA based)")
    group.add_argument(
        "--dim-reducer",
        type=str,
        default="None",
        choices=["None", "PCA", "KernelPCA"],
        help="Dimensionality reduction method to use. Default: %(default)s",
    )
    group.add_argument(
        "--ndim-reduced",
        type=str,
        default=None,
        help="Number of dimensions to reduce the embeddings to. Can be either 'mle' or an integer.",
    )
    group.add_argument(
        "--pca-variance",
        type=float,
        default=None,
        help=(
            "Select the number of dimensions to use based on a target fraction of"
            " the total variance explained by the retained dimensions."
        ),
    )
    group.add_argument(
        "--pca-kernel",
        type=str,
        default="linear",
        choices=["linear", "poly", "rbf", "sigmoid", "cosine"],
        help="PCA kernel to use. Default: %(default)s",
    )
    group = parser.add_argument_group(
        "Dimensionality reduction (nearest-neighbour/manifold based)"
    )
    group.add_argument(
        "--dim-reducer-man",
        type=str,
        default="None",
        choices=["None", "UMAP", "tSNE"],
        help="Manifold dimensionality reduction method to use. Default: %(default)s",
    )
    group.add_argument(
        "--ndim-reduced-man",
        type=int,
        default=None,
        help="Number of dimensions to reduce the embeddings to.",
    )
    group.add_argument(
        "--dim-reducer-man-metric",
        type=str,
        default="euclidean",
        choices=METRICS,
        help="Distance metric for manifold dimensionality reduction. Default: %(default)s",
    )

    # TODO Add arguments for Kernel PCA kernels

    # Normalization Args
    group = parser.add_argument_group("Normalization")
    mx_group = group.add_mutually_exclusive_group()
    mx_group.add_argument(
        "--zscore",
        dest="zscore",
        action="store_true",
        default=None,
        help=(
            "Standardize with the z-score of each dimension (applied before reduction)."
            " Default: True if using PCA, False otherwise."
        ),
    )
    mx_group.add_argument(
        "--no-zscore",
        dest="zscore",
        action="store_false",
        default=None,
        help="Don't standardize data as the z-score of each dimension before PCA.",
    )
    group.add_argument(
        "--normalize",
        action="store_true",
        help="L2 normalize embeddings (After PCA if applicable)",
    )
    mx_group = group.add_mutually_exclusive_group()
    mx_group.add_argument(
        "--zscore2",
        dest="zscore2",
        action="store_true",
        default=False,
        help=(
            "Standardize with the z-score of each dimension, and divide by sqrt(ndim)."
            " (Applied after reduction, before clustering)."
            " Default: disabled."
        ),
    )
    mx_group.add_argument(
        "--no-zscore2",
        dest="zscore2",
        action="store_false",
        default=False,
        help=(
            "Don't standardize data as the z-score of each dimension between"
            " reduction and clustering."
            " This is the default behaviour."
        ),
    )

    # Clusterer args ----------------------------------------------------------
    group = parser.add_argument_group("Clustering")
    group.add_argument(
        "--clusterer-name",
        type=str,
        default="HDBSCAN",
        choices=CLUSTERERS,
        help="Name of clustering method. Default: %(default)s",
    )
    group.add_argument(
        "--distance-metric",
        type=str,
        default="euclidean",
        choices=METRICS,
        help="Metric function for clustering methods",
    )
    group.add_argument(
        "--max-iter",
        type=int,
        default=1_000,
        help="Maximum number of iterations for iterative clusterers. Default: %(default)s",
    )
    group.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum number of samples to comprise a cluster. Default: %(default)s",
    )
    group.add_argument(
        "--max-samples",
        type=float,
        help=(
            "HDBSCAN's maximum number of samples to comprise a cluster. This can be"
            " either an integer >= 2 (the maximum number of samples per cluster), or a"
            " float 0 < MAX_SAMPLES < 1, in which case this is the maximum fraction of"
            " the dataset that can be in a single cluster. Default: infinity."
        ),
    )
    group.add_argument(
        "--affinity-damping",
        type=float,
        default=0.5,
        help=(
            "Affinity propagation's damping factor in the range [0.5, 1.0)."
            " Default: %(default)s"
        ),
    )
    group.add_argument(
        "--affinity-conv-iter",
        type=int,
        default=15,
        help=(
            "Affinity propagation's stopping criteria, number of iterations with"
            " no change in number of estimated clusters. Default: %(default)s"
        ),
    )
    group.add_argument(
        "--spectral-assigner",
        type=str,
        default="cluster_qr",
        choices=["kmeans", "discretize", "cluster_qr"],
        help="Spectral clustering label assignment method. Default: %(default)s",
    )
    group.add_argument(
        "--aggclust-linkage",
        type=str,
        default="ward",
        choices=["ward", "complete", "average", "single"],
        help="Linkage method for agglomerative clustering method",
    )
    group.add_argument(
        "--aggclust-dist-thresh",
        type=float,
        help=(
            "Distance threshold for agglomerative clustering method. If unset,"
            " the true number of clusters will be given to the clusterer"
            " instead of using a distance threshold."
        ),
    )
    group.add_argument(
        "--hdbscan-method",
        type=str,
        default="eom",
        choices=["eom", "leaf"],
        help="HDBSCAN cluster extraction method. Default: %(default)s",
    )
    group.add_argument(
        "--optics-method",
        type=str,
        default="xi",
        choices=["xi", "dbscan"],
        help="OPTICS cluster extraction method. Default: %(default)s",
    )
    group.add_argument(
        "--optics-xi",
        type=float,
        default=0.05,
        help=(
            "OPTICS minimum steepness, xi. Only applies when using the"
            " xi cluster method for OPTICS. Default: %(default)s"
        ),
    )

    # Hardware configuration args ---------------------------------------------
    group = parser.add_argument_group("Hardware configuration")
    group.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Number of CPU workers to use. Default: number of CPU cores available.",
    )

    # Logging args ------------------------------------------------------------
    group = parser.add_argument_group("Debugging and logging")
    group.add_argument(
        "--log-wandb",
        action="store_true",
        help="Log results with Weights & Biases https://wandb.ai",
    )
    group.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Overrides --log-wandb and ensures wandb is always disabled.",
    )
    group.add_argument(
        "--wandb-entity",
        type=str,
        default="uoguelph_mlrg",
        help=(
            "The entity (organization) within which your wandb project is"
            " located. Default: %(default)s"
        ),
    )
    group.add_argument(
        "--wandb-project",
        type=str,
        default="zs-ssl-clustering",
        help="Name of project on wandb, where these runs will be saved. Default: %(default)s",
    )
    group.add_argument(
        "--wandb-tags",
        nargs="+",
        type=str,
        help="Tag(s) to add to wandb run. Multiple tags can be given, separated by spaces.",
    )
    group.add_argument(
        "--wandb-group",
        type=str,
        default="",
        help="Used to group wandb runs together, to run stats on them together.",
    )
    group.add_argument(
        "--run-name",
        type=str,
        help="Human-readable identifier for the model run or job. Used to name the run on wandb.",
    )
    group.add_argument(
        "--run-id",
        type=str,
        help="Unique identifier for the model run or job. Used as the run ID on wandb.",
    )
    # Verbosity args ----------------------------------------------------------
    group = parser.add_argument_group("Verbosity")
    group.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="Increase the level of verbosity. The default verbosity level is %(default)s.",
    )
    group.add_argument(
        "--quiet",
        "-q",
        action="count",
        default=0,
        help="Decrease the level of verbosity.",
    )

    return parser


def cli():
    r"""Command-line interface for model training."""
    parser = get_parser()
    config = parser.parse_args()
    # Handle disable_wandb overriding log_wandb and forcing it to be disabled.
    if config.disable_wandb:
        config.log_wandb = False
    del config.disable_wandb
    # Handle unspecified wandb_tags: if None, set to empty list
    if config.wandb_tags is None:
        config.wandb_tags = []
    config.verbose -= config.quiet
    del config.quiet
    return run(config)


if __name__ == "__main__":
    cli()
