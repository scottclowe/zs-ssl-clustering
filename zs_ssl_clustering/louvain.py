"""
Louvain community detection algorithm.
"""

import networkx as nx
import numpy as np
import sklearn.metrics
from sklearn.base import BaseEstimator, ClusterMixin


def build_affinity_matrix(
    XA, XB=None, metric="l2", remove_self_loops=True, n_jobs=None
):
    """
    Build an affinity matrix from pairwise distances.

    Parameters
    ----------
    XA : ndarray of shape (n_samples, n_features)
        The input data.

    XB : ndarray of shape (n_samples, n_features), optional
        The second input data. If not provided, XA is used as the second input data.

    metric : str, default="l2"
        The distance metric to use.

    remove_self_loops : bool, default=True
        Whether to remove self-loops from the affinity matrix.

    n_jobs : int or None, optional
        The number of parallel jobs to run for computing pairwise distances.
        Default is ``None``.

    Returns
    -------
    ndarray of shape (n_samples, n_samples)
        The affinity matrix.
    """
    if XB is None:
        XB = XA
    dst = sklearn.metrics.pairwise_distances(XA, XB, metric=metric, n_jobs=n_jobs)
    if metric == "cosine":
        # Range [-1, 1] from maximally dissimilar to maximally similar.
        dst = (dst + 1) / 2
        # Range [0, 1] from maximally dissimilar to maximally similar.
    else:
        # Range [0, infty] from maximally similar to maximally dissimilar.
        dst = 1 / (1 + dst)
        # Range [0, 1] from maximally dissimilar to maximally similar.
    if remove_self_loops:
        # Self loops represented already reduced communities, large values indicate strong communities. Remove them.
        np.fill_diagonal(dst, 0)
    return dst


class LouvainCommunities(ClusterMixin, BaseEstimator):
    """
    Louvain community detection algorithm.

    Parameters
    ----------
    metric : str, default="l2"
        The metric to use when calculating pairwise distances between samples.
        See :func:`sklearn.metrics.pairwise_distances` for valid options.

    resolution : float, default=1.0
        If resolution is less than 1, the algorithm favors larger communities.
        Greater than 1 favors smaller communities.

    threshold : float, default=1e-07
        Modularity gain threshold for each level. If the gain of modularity
        between 2 levels of the algorithm is less than the given threshold then
        the algorithm stops and returns the resulting communities.

    remove_self_loops : bool, default=True
        Whether to remove self-loops from the adjacency matrix.

    seed : int, default=None
        Random seed for the algorithm.

    n_jobs : int, default=None
        Number of jobs to run in parallel when measuring the pairwise distance
        between samples. If -1, then the number of jobs is set to the number
        of CPU cores.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        Stores the affinity matrix used in ``fit``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.
    """

    def __init__(
        self,
        metric="l2",
        resolution=1.0,
        threshold=1e-07,
        remove_self_loops=True,
        seed=None,
        n_jobs=None,
    ):
        self.metric = metric
        self.resolution = resolution
        self.threshold = threshold
        self.remove_self_loops = remove_self_loops
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        # Create adjacency matrix from pairwise distances
        dst = build_affinity_matrix(
            X,
            metric=self.metric,
            remove_self_loops=self.remove_self_loops,
            n_jobs=self.n_jobs,
        )
        self.affinity_matrix_ = dst
        # Convert to NetworkX graph
        G = nx.from_numpy_array(dst)
        # Partition into Louvain communities
        partition = nx.community.louvain_communities(
            G, resolution=self.resolution, threshold=self.threshold, seed=self.seed
        )
        # Convert partitions to a label vector
        labels = np.zeros(dst.shape[-1])
        for idx in range(len(partition)):
            community = list(partition[idx])
            labels[community] = idx
        # Set labels_ attribute
        self.labels_ = labels
        return self
