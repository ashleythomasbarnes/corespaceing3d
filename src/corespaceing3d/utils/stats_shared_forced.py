"""Helpers for shared/forced edge-length comparisons."""

import numpy as np


__all__ = [
    "_edge_sets",
    "build_shared_arrays",
    "build_shared_arrays_full",
    "build_forced_arrays",
]


def _empty_float() -> np.ndarray:
    return np.array([], dtype=float)


def _empty_int() -> np.ndarray:
    return np.array([], dtype=int)

def _edge_sets(edges3d, edges2d):
    """Return (shared, only3, only2, union) as sets of unordered node pairs."""
    s3 = {frozenset((i, j)) for i, j, _ in edges3d}
    s2 = {frozenset((i, j)) for i, j, _ in edges2d}
    shared = s3 & s2
    only3 = s3 - s2
    only2 = s2 - s3
    union = s3 | s2
    return shared, only3, only2, union


def _pair_index_arrays(pairs):
    if not pairs:
        return _empty_int(), _empty_int()
    ij = [tuple(e) for e in pairs]
    i = np.fromiter((min(a, b) for a, b in ij), dtype=int, count=len(ij))
    j = np.fromiter((max(a, b) for a, b in ij), dtype=int, count=len(ij))
    return i, j


def _pair_distance_arrays(pts2d, pts3d, pairs):
    """For a set of unordered node pairs, return d2 (2D) and d3 (3D) arrays."""
    if not pairs:
        return _empty_float(), _empty_float()
    i, j = _pair_index_arrays(pairs)
    d2 = np.linalg.norm(pts2d[i] - pts2d[j], axis=1)
    d3 = np.linalg.norm(pts3d[i] - pts3d[j], axis=1)
    return d2, d3


def _build_stats(d2, d3):
    if d2.size == 0:
        empty = _empty_float()
        return {"d2": empty, "d3": empty, "diff": empty, "ratio": empty, "count": 0}
    return {
        "d2": d2,
        "d3": d3,
        "diff": d2 - d3,
        "ratio": d2 / d3,
        "count": d2.size,
    }


def build_shared_arrays(pts3d, pts2d, edges3d, edges2d):
    """Return dict with shared-only arrays: d2, d3, diff (d2-d3), ratio (d2/d3), count."""
    return build_shared_arrays_full(pts3d, pts2d, edges3d, edges2d)

def build_shared_arrays_full(pts3d, pts2d, edges3d, edges2d):
    """Shared pairs with both 2D and 3D lengths and ratios (like your build_shared_arrays, but explicit)."""
    shared, _, _, _ = _edge_sets(edges3d, edges2d)
    d2_s, d3_s = _pair_distance_arrays(pts2d, pts3d, shared)
    return _build_stats(d2_s, d3_s)

def build_forced_arrays(pts3d, pts2d, edges3d):
    """For all 3D NN edges, return d3 (as in edges3d), d2 (projected), diff, ratio."""
    if not edges3d:
        return _build_stats(_empty_float(), _empty_float())

    i3 = np.fromiter((min(i, j) for (i, j, _) in edges3d), dtype=int, count=len(edges3d))
    j3 = np.fromiter((max(i, j) for (i, j, _) in edges3d), dtype=int, count=len(edges3d))
    d2 = np.linalg.norm(pts2d[i3] - pts2d[j3], axis=1)
    d3 = np.fromiter((w for (_, _, w) in edges3d), dtype=float, count=len(edges3d))
    return _build_stats(d2, d3)
