import os
import time
from datetime import datetime

import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import adjusted_rand_score as ARAND
from sklearn.metrics import silhouette_score as SIL

from zs_ssl_clustering import io


def run(config):
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

    data = np.load(io.get_embeddings_path(config))
    embeddings = data["embeddings"]
    y_true = data["y_true"]
    n_clusters_gt = len(np.unique(y_true))

    if config.pca_method != "None":
        use_kernel_PCA = config.pca_method == "KernelPCA"
        pca_components = config.pca_components
        pca_variance = config.pca_variance

        if pca_components is None and pca_variance is None:
            raise ValueError(
                "Neither 'pca_components' nor 'pca_variance' was specified"
            )
        elif pca_components is not None and pca_variance is not None:
            raise ValueError("Both 'pca_components' and 'pca_variance' was specified")

        if pca_components is not None:
            assert isinstance(pca_components, int), "pca_components must be int"
            n_components = pca_components
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

        shape_before = embeddings.shape[-1]

        start_pca = time.time()
        if use_kernel_PCA:
            pca = KernelPCA(
                n_components=n_components, kernel="linear", random_state=config.seed
            )
        else:
            pca = PCA(n_components=n_components, random_state=config.seed)
        embeddings = pca.fit_transform(embeddings)
        end_pca = time.time()

        shape_after = embeddings.shape[-1]

        print(f"Shape Before/After PCA: {shape_before}/{shape_after}")
        if not use_kernel_PCA:
            print(
                f"PCA Explained Variance: {np.sum(pca.explained_variance_ratio_)*100} %"
            )
        print(f"PCA Fitting time: {end_pca-start_pca:.2f}s")

    # TODO: Maybe do before PCA?
    if config.normalize:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    cluster_method = config.cluster_method
    if cluster_method == "AgglomerativeClustering":
        # Can work with specified number of clusters, as well as unknown (which requires a distance threshold)
        # We can also impose some structure metric through the "connectivity" argument
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric=config.aggclust_metric,
            linkage=config.aggclust_linkage,
            distance_threshold=config.aggclust_dist_thresh,
        )
    elif cluster_method == "HDBSCAN":
        clusterer = HDBSCAN(min_cluster_size=2)
    elif cluster_method == "KMeans":
        clusterer = KMeans(n_clusters=n_clusters_gt, random_state=config.seed)
    elif cluster_method == "Spectral":
        # TODO Look into this:
        # Requires the number of clusters
        # Can be estimated through e.g.
        # https://proceedings.neurips.cc/paper_files/paper/2004/file/40173ea48d9567f1f393b20c855bb40b-Paper.pdf
        # Might be more recent work to consider
        clusterer = SpectralClustering(
            n_clusters=n_clusters_gt, random_state=config.seed
        )

    start_cluster = time.time()
    clusterer.fit(embeddings)
    end_cluster = time.time()

    y_pred = clusterer.labels_
    n_clusters_pred = len(np.unique(y_pred))
    ami_score = AMI(y_true, y_pred)
    arand_score = ARAND(y_true, y_pred)
    silhouette_score = SIL(embeddings, y_true, metric="euclidean")

    print(f"Cluster Fitting time: {end_cluster-start_cluster:.2f}s")
    print(f"Number of clusters GT\t{n_clusters_gt}")
    print(f"Number of clusters Pred\t{n_clusters_pred}")
    print(f"AMI\t{ami_score}")
    print(f"ARand\t{arand_score}")
    print(f"Silhouette\t{silhouette_score}")

    if config.log_wandb:
        wandb.log(
            {
                "AMI": ami_score,
                "ARand": arand_score,
                "Silhouette": silhouette_score,
                "Clusters_pred": n_clusters_pred,
                "Clusters_gt": n_clusters_gt,
                "Fitting_time": end_cluster - start_cluster,
            }
        )


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

    # PCA Args
    group = parser.add_argument_group("PCA Arguments")
    group.add_argument(
        "--pca-method",
        type=str,
        default="None",
        choices=["None", "PCA", "KernelPCA"],
        help="Toggle for using PCA",
    )
    group.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help="Number of Principle Components to keep",
    )
    group.add_argument(
        "--pca-variance",
        type=float,
        default=None,
        help="Percentage of Variance to be explained by Principle Components",
    )

    # TODO Add arguments for Kernel PCA kernels

    # Normalization Args
    group = parser.add_argument_group("Normalization")
    group.add_argument(
        "--normalize",
        action="store_true",
        help="L2 normalize embeddings (After PCA if applicable)",
    )

    # Clustering Args
    group = parser.add_argument_group("Clustering")
    group.add_argument(
        "--cluster-method",
        type=str,
        default="KMeans",
        choices=["KMeans", "HDBSCAN", "AgglomerativeClustering", "SpectralClustering"],
        help="Method to cluster embeddings with",
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
        "--aggclust-metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "manhattan", "cosine"],
        help="Metric function for agglomerative clustering method",
    )

    # TODO Add more arguments for clustering

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
        help=(
            "The entity (organization) within which your wandb project is"
            ' located. By default, this will be your "default location" set on'
            " wandb at https://wandb.ai/settings"
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
    return run(config)


if __name__ == "__main__":
    cli()
