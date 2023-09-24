#!/usr/bin/env python

import os
import time
from datetime import datetime

import numpy as np
import sklearn.cluster
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


def run(config):
    print("Configuration:")
    print()
    print(config)
    print()

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

    clusterer_args_used = set()

    start_reducing = time.time()
    if config.dim_reducer == "None":
        pass
    elif "PCA" in config.dim_reducer:
        use_kernel_PCA = config.dim_reducer == "KernelPCA"
        ndim_reduced = config.ndim_reduced
        pca_variance = config.pca_variance

        if ndim_reduced is None and pca_variance is None:
            raise ValueError("Neither 'ndim_reduced' nor 'pca_variance' was specified")
        elif ndim_reduced is not None and pca_variance is not None:
            raise ValueError("Both 'ndim_reduced' and 'pca_variance' was specified")

        if ndim_reduced is not None:
            assert isinstance(ndim_reduced, int), "ndim_reduced must be int"
            n_components = ndim_reduced
        else:
            if use_kernel_PCA:
                raise ValueError(
                    "Cannot use KernelPCA by specifying the variance to be kept"
                )
            assert isinstance(pca_variance, float), "pca_variance must be float"
            assert (
                pca_variance > 0.0 and pca_variance < 1.0
            ), "pca_variance must be between 0 and 1"
            n_components = pca_variance

        # Standardize to zero mean, unit variance
        embeddings = (embeddings - np.mean(embeddings, axis=0)) / np.std(
            embeddings, axis=0
        )

        start_pca = time.time()
        if use_kernel_PCA:
            pca = KernelPCA(
                n_components=n_components,
                kernel="linear",
                random_state=config.seed,
                n_jobs=config.workers,
            )
            clusterer_args_used.add("workers")
        else:
            pca = PCA(n_components=n_components, random_state=config.seed)
        embeddings = pca.fit_transform(embeddings)
        end_pca = time.time()

        shape_after = embeddings.shape[-1]

        print(f"Shape Before/After PCA: {encoding_dim}/{shape_after}")
        if not use_kernel_PCA:
            print(
                f"PCA Explained Variance: {np.sum(pca.explained_variance_ratio_)*100} %"
            )
        print(f"PCA Fitting time: {end_pca-start_pca:.2f}s")
    else:
        raise ValueError(
            f"Unrecognized dimensionality reduction method: '{config.dim_reducer}'"
        )
    end_reducing = time.time()

    # TODO: Maybe do before PCA?
    if config.normalize:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    reduced_dim = embeddings.shape[-1]

    clusterer_args = {
        "distance_metric",
        "max_iter",
        "min_samples",
        "workers",
        "aggclust_linkage",
        "aggclust_dist_thresh",
        "affinity_damping",
        "affinity_conv_iter",
        "optics_method",
        "optics_xi",
    }

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
            random_state=config.seed,
            verbose=config.verbose > 0,
        )
        clusterer_args_used = clusterer_args_used.union({"seed"})

    elif config.clusterer_name == "AgglomerativeClustering":
        # Can work with specified number of clusters, as well as unknown (which requires a distance threshold)
        # We can also impose some structure metric through the "connectivity" argument
        clusterer = sklearn.cluster.AgglomerativeClustering(
            n_clusters=None,
            metric=config.distance_metric,
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
        clusterer = sklearn.cluster.HDBSCAN(
            min_cluster_size=config.min_samples,
            metric=config.distance_metric,
            n_jobs=config.workers,
        )
        clusterer_args_used = clusterer_args_used.union(
            {
                "min_samples",
                "distance_metric",
                "workers",
            }
        )

    elif config.clusterer_name == "OPTICS":
        clusterer = sklearn.cluster.OPTICS(
            min_samples=config.min_samples,
            metric=config.distance_metric,
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

    # Wipe the state of cluster arguments that were not relevant to the
    # chosen clusterer.
    clusterer_args_unused = clusterer_args.difference(clusterer_args_used)
    for key in clusterer_args_unused:
        if key == "distance_metric":
            continue
        setattr(config, key, None)
        if config.log_wandb:
            wandb.config.update({key: None}, allow_val_change=True)
    if config.log_wandb and config.workers is not None and config.workers == -1:
        wandb.config.update(
            {"workers": utils.get_num_cpu_available()}, allow_val_change=True
        )

    start_cluster = time.time()
    clusterer.fit(embeddings)
    end_cluster = time.time()

    y_pred = clusterer.labels_
    n_clusters_pred = sum(np.unique(y_pred) >= 0)

    ratio_clustered = np.sum(y_pred >= 0) / len(y_pred)
    ratio_unclustered = 1 - ratio_clustered
    results = {
        "n_samples": len(embeddings),
        "encoding_dim": encoding_dim,
        "reduced_dim": reduced_dim,
        "time_reducing": end_reducing - start_reducing,
        "time_clustering": end_cluster - start_cluster,
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
        "DBS_true": sklearn.metrics.davies_bouldin_score(embeddings, y_true),
        "silhouette_true": sklearn.metrics.silhouette_score(
            embeddings, y_true, metric="euclidean"
        ),
    }
    if config.distance_metric != "euclidean":
        results[f"silhouette-{config.distance_metric}_true"] = (
            sklearn.metrics.silhouette_score(
                embeddings, y_true, metric=config.distance_metric
            ),
        )

    if n_clusters_pred > 1 and n_clusters_pred < len(embeddings):
        results["CHS_pred"] = sklearn.metrics.calinski_harabasz_score(
            embeddings, y_pred
        )
        results["DBS_pred"] = sklearn.metrics.davies_bouldin_score(embeddings, y_pred)
        results["silhouette_pred"] = sklearn.metrics.silhouette_score(
            embeddings, y_pred, metric="euclidean"
        )
        if config.distance_metric != "euclidean":
            results[f"silhouette-{config.distance_metric}_pred"] = (
                sklearn.metrics.silhouette_score(
                    embeddings, y_pred, metric=config.distance_metric
                ),
            )

    if hasattr(clusterer, "n_iter_"):
        results["iter"] = clusterer.n_iter_  # Number of iterations run.
        results["converged"] = clusterer.n_iter_ < config.max_iter

    print(
        f"\n{config.clusterer_name}({config.model}({config.dataset_name}))"
        " evaluation results:"
    )
    for k, v in results.items():
        if k == "time_clustering":
            print(f"  {k + ' ':.<24s} {v:.4f} seconds")
        elif isinstance(k, int):
            print(f"  {k + ' ':.<24s} {v:>3d}")
        else:
            print(f"  {k + ' ':.<24s} {v:.4f}")

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
        default=42,
        help="Random number generator (RNG) seed. Default: not controlled",
    )
    # Dimensionality reduction args -------------------------------------------
    group = parser.add_argument_group("Dimensionality reduction")
    group.add_argument(
        "--dim-reducer",
        type=str,
        default="None",
        choices=["None", "PCA", "KernelPCA"],
        help="Dimensionality reduction method to use. Default: %(default)s",
    )
    group.add_argument(
        "--ndim-reduced",
        type=int,
        default=None,
        help="Number of dimensions to reduce the embeddings to.",
    )
    group.add_argument(
        "--pca-variance",
        type=float,
        default=None,
        help=(
            "Select the number of dimensions to use based on a target fraction of"
            " the total variance explained by the retained dimensions.",
        ),
    )

    # TODO Add arguments for Kernel PCA kernels

    # Normalization Args
    group = parser.add_argument_group("Normalization")
    group.add_argument(
        "--normalize",
        action="store_true",
        help="L2 normalize embeddings (After PCA if applicable)",
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
        choices=list(sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS.keys()),
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
        default=2,
        help="Minimum number of samples to comprise a cluster. Default: %(default)s",
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
        "--aggclust-linkage",
        type=str,
        default="ward",
        choices=["ward", "complete", "average", "single"],
        help="Linkage method for agglomerative clustering method",
    )
    group.add_argument(
        "--aggclust-dist-thresh",
        type=float,
        default=1,
        help="Distance threshold for agglomerative clustering method",
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
            " located. Default: %(default)s",
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
