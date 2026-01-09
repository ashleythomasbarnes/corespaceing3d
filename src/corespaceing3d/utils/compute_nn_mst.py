"""Nearest-neighbour and minimum-spanning-tree utilities.

This module provides helpers to compute:
- the 1-nearest-neighbour (NN) graph (collapsed to unique undirected edges), and
- the Euclidean minimum spanning tree (MST),

returning both edge lists and summary statistics of the edge-length
distribution.

Notes
-----
- Distances are computed with SciPy pairwise distances.
- The MST is computed with :func:`scipy.sparse.csgraph.minimum_spanning_tree`.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

from .print_stats import summarize_lengths


__all__ = [
    "compute_nn",
    "compute_mst",
]


def _empty_result() -> tuple[float, list[tuple[int, int, float]], set[frozenset[int]], np.ndarray, dict]:
    lengths = np.array([], dtype=float)
    return 0.0, [], set(), lengths, summarize_lengths(lengths)


def compute_nn(points: np.ndarray) -> tuple[float, list[tuple[int, int, float]], set[frozenset[int]], np.ndarray, dict]:
    """Compute the 1-nearest-neighbour graph for a point set.

    The NN graph is first formed as a directed graph (each node links to its
    closest neighbour), then collapsed to unique undirected edges.

    Parameters
    ----------
    points : (N, D) ndarray
        Point coordinates.

    Returns
    -------
    lengths_sum : float
        Total NN graph length (sum of unique undirected edge lengths).
    edges_list : list of (i, j, length)
        Unique undirected edges with ``i < j``.
    edge_set : set of frozenset
        Set of edges as ``frozenset({i, j})`` for convenient comparisons.
    lengths : (M,) ndarray
        Edge lengths corresponding to ``edges_list``.
    stats : dict
        Summary statistics for ``lengths`` (from :func:`utils.print_stats.summarize_lengths`).

    Notes
    -----
    This procedure produces at most ``N`` unique edges (unlike an MST, which has
    exactly ``N-1`` edges for ``N >= 2``).
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        return _empty_result()

    # Pairwise distances
    D = squareform(pdist(points))
    np.fill_diagonal(D, np.inf)

    # Directed nearest neighbour for each i
    nn_j = np.argmin(D, axis=1)          # (N,)
    nn_d = D[np.arange(D.shape[0]), nn_j]  # (N,)

    # Collapse to unique undirected edges: keep (min(i,j), max(i,j)) with length d(i,j)
    edges_map = {}
    for i, j, d in zip(range(D.shape[0]), nn_j, nn_d):
        a, b = (i, j) if i < j else (j, i)
        # If both directions exist, lengths are equal; keep one
        if (a, b) not in edges_map or d < edges_map[(a, b)]:
            edges_map[(a, b)] = float(d)

    edges_list = [(i, j, w) for (i, j), w in edges_map.items()]
    edge_set = {frozenset((i, j)) for (i, j, _) in edges_list}

    lengths = np.array([w for (_, _, w) in edges_list], dtype=float)
    lengths_sum = float(lengths.sum())

    stats = summarize_lengths(lengths)

    return lengths_sum, edges_list, edge_set, lengths, stats


def compute_mst(points: np.ndarray) -> tuple[float, list[tuple[int, int, float]], set[frozenset[int]], np.ndarray, dict]:
    """Compute the Euclidean minimum spanning tree for a point set.

    Parameters
    ----------
    points : (N, D) ndarray
        Point coordinates.

    Returns
    -------
    lengths_sum : float
        Total MST length.
    edges_list : list of (i, j, length)
        MST edges with ``i < j``.
    edge_set : set of frozenset
        Set of edges as ``frozenset({i, j})`` for convenient comparisons.
    lengths : (N-1,) ndarray
        MST edge lengths.
    stats : dict
        Summary statistics for ``lengths`` (from :func:`utils.print_stats.summarize_lengths`).
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        return _empty_result()

    D = squareform(pdist(points))           # NxN dense distance matrix
    mst_sparse = minimum_spanning_tree(D)   # CSR, directed

    # Extract edges from sparse matrix
    ii, jj = mst_sparse.nonzero()
    lengths = np.asarray(mst_sparse[ii, jj]).ravel()

    edges = []
    for a, b, w in zip(ii, jj, lengths):
        i, j = (a, b) if a < b else (b, a)
        edges.append((i, j, float(w)))

    # Deduplicate (should already be N-1 unique edges)
    edges_unique = {(i, j): w for i, j, w in edges}
    edges_list = [(i, j, w) for (i, j), w in edges_unique.items()]
    edge_set = {frozenset((i, j)) for (i, j, _) in edges_list}

    # Rebuild lengths to match deduped list order
    lengths = np.array([w for (_, _, w) in edges_list], dtype=float)
    lengths_sum = float(lengths.sum())

    # Summary stats
    stats = summarize_lengths(lengths)

    return lengths_sum, edges_list, edge_set, lengths, stats
