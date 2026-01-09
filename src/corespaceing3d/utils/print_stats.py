"""Statistics and reporting utilities.

This module centralises lightweight utilities for:

- summarising 1D length arrays,
- comparing 3D vs 2D edge sets (NN/MST style edge lists),
- building a rich “projection summary” payload used for analysis and paper text,
- printing user-friendly reports.

Notes
-----
- This module intentionally contains no sampling or plotting code.
- Edge lists are assumed to contain tuples ``(i, j, w)`` where ``i`` and ``j``
  are node indices and ``w`` is a (positive) edge length.
"""

from __future__ import annotations

import json

import numpy as np

__all__ = [
    "summarize_lengths",
    "print_length_stats",
    "compare_edge_sets",
    "compare_nns",
    "compare_msts",
    "compute_nn_projection_summary",
    "print_projection_summary",
    "print_nn_report",
]


# -----------------------------------------------------------------------------
# Core array statistics
# -----------------------------------------------------------------------------

def summarize_lengths(x: np.ndarray) -> dict:
    """Return summary stats for a 1D array of positive edge lengths.

    Parameters
    ----------
    x : (N,) ndarray
        Input values.

    Returns
    -------
    stats : dict
        Dictionary containing count, location/scale estimators, and selected
        percentiles.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "mad": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
        }

    med = np.median(x)
    mad = np.median(np.abs(x - med))  # (unscaled) median absolute deviation
    q = np.percentile(x, [10, 25, 50, 75, 90, 95, 99])

    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(med),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "mad": float(mad),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p10": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p90": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
    }


def print_length_stats(label: str, s: dict) -> None:
    """Pretty-print summary statistics from :func:`summarize_lengths`."""
    print(f"\n{label} edge-length stats (N={s['n']}):")
    print(
        "  "
        f"mean={s['mean']:.5f}, median={s['median']:.5f}, "
        f"std={s['std']:.5f}, mad={s['mad']:.5f}"
    )
    print(
        "  "
        f"min={s['min']:.5f}, p10={s['p10']:.5f}, p25={s['p25']:.5f}, "
        f"p50={s['p50']:.5f}, p75={s['p75']:.5f}, p90={s['p90']:.5f}, "
        f"p95={s['p95']:.5f}, p99={s['p99']:.5f}, max={s['max']:.5f}"
    )


# -----------------------------------------------------------------------------
# Edge-set comparisons
# -----------------------------------------------------------------------------

def compare_edge_sets(edges3d, edges2d) -> dict:
    """Compare two edge lists by node pairs.

    Parameters
    ----------
    edges3d, edges2d : list of (i, j, w)
        Edge lists.

    Returns
    -------
    out : dict
        Dictionary with counts, overlap metrics, and the shared/unique pair sets.

    Notes
    -----
    Pairs are compared as unordered node sets ``frozenset({i, j})``.
    """
    set3 = {frozenset((i, j)) for (i, j, _) in edges3d}
    set2 = {frozenset((i, j)) for (i, j, _) in edges2d}

    shared = set3 & set2
    only3 = set3 - set2
    only2 = set2 - set3

    union = set3 | set2
    jaccard = len(shared) / len(union) if union else 1.0
    overlap_fraction = len(shared) / len(set3) if set3 else 1.0

    return {
        "n_shared_edges": len(shared),
        "n_only_in_3d": len(only3),
        "n_only_in_2d": len(only2),
        "overlap_fraction_3d": overlap_fraction,  # fraction of 3D edges recovered in 2D
        "jaccard_similarity": jaccard,
        "shared_edges": shared,
        "only_in_3d_edges": only3,
        "only_in_2d_edges": only2,
    }


def compare_msts(edges3d, edges2d) -> dict:
    """Alias of :func:`compare_edge_sets` for MST edge lists."""
    return compare_edge_sets(edges3d, edges2d)


def compare_nns(edges3d, edges2d) -> dict:
    """Alias of :func:`compare_edge_sets` for NN edge lists."""
    return compare_edge_sets(edges3d, edges2d)


# -----------------------------------------------------------------------------
# Projection summary helpers
# -----------------------------------------------------------------------------

def _edge_sets(edges3d, edges2d):
    s3 = {frozenset((i, j)) for i, j, _ in edges3d}
    s2 = {frozenset((i, j)) for i, j, _ in edges2d}
    shared = s3 & s2
    only3 = s3 - s2
    only2 = s2 - s3
    union = s3 | s2
    return shared, only3, only2, union


def _pairs_to_arrays(pairs):
    """Return arrays i, j (i<j) from a set/list of unordered pairs."""
    if not pairs:
        return np.array([], dtype=int), np.array([], dtype=int)
    ij = [tuple(e) for e in pairs]
    i = np.fromiter((min(a, b) for a, b in ij), dtype=int, count=len(ij))
    j = np.fromiter((max(a, b) for a, b in ij), dtype=int, count=len(ij))
    return i, j


def _edge_lengths_from_pairs(pts, pairs):
    """Compute Euclidean lengths for given unordered pairs using coordinates ``pts``."""
    i, j = _pairs_to_arrays(pairs)
    if i.size == 0:
        return np.array([], dtype=float)
    return np.linalg.norm(pts[i] - pts[j], axis=1)


def _edges_to_adjacency(edges, n_nodes):
    """Convert an edge list to an undirected adjacency list."""
    adj = [set() for _ in range(n_nodes)]
    for i, j, _ in edges:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _stats_1d(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p90": np.nan,
        }
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "p10": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p90": float(q[4]),
    }


def _round_or_nan(x, nd=3):
    try:
        return None if (x is None or np.isnan(x)) else float(np.round(x, nd))
    except Exception:
        return None


def compute_nn_projection_summary(pts3d, pts2d, edges3d, edges2d):
    """Build a rich summary dict for 3D vs 2D NN/MST comparisons.

    This supports both analysis plots and generation of paper-ready text.

    Returns
    -------
    summary : dict
        Contains counts, overlap metrics, length statistics for shared/unique
        edges, forced measurements, reassignment diagnostics, and a rounded
        ``paper_text_payload``.
    """
    n_nodes = pts3d.shape[0]
    geom_pi_over_4 = np.pi / 4.0

    shared, only3, only2, union = _edge_sets(edges3d, edges2d)

    # ---- Shared pairs (exist in both edge sets) ----
    d3_shared = _edge_lengths_from_pairs(pts3d, shared)
    d2_shared = _edge_lengths_from_pairs(pts2d, shared)
    d_shared = d2_shared - d3_shared
    r_shared = np.divide(
        d2_shared,
        d3_shared,
        out=np.full_like(d2_shared, np.nan, dtype=float),
        where=d3_shared > 0,
    )

    # ---- One-sided pairs, also measured in the other space for diagnostics ----
    d3_only3 = _edge_lengths_from_pairs(pts3d, only3)          # native 3D
    d2_proj_only3 = _edge_lengths_from_pairs(pts2d, only3)     # projected to 2D
    d2_only2 = _edge_lengths_from_pairs(pts2d, only2)          # native 2D
    d3_back_only2 = _edge_lengths_from_pairs(pts3d, only2)     # back-projected to 3D

    # ---- Connectivity overlap: identical neighbor sets (strict) ----
    adj3 = _edges_to_adjacency(edges3d, n_nodes)
    adj2 = _edges_to_adjacency(edges2d, n_nodes)
    identical_nodes = sum(1 for i in range(n_nodes) if adj3[i] == adj2[i])
    frac_identical_nodes = identical_nodes / n_nodes if n_nodes else np.nan

    # ---- Overlap / Jaccard ----
    n_shared = len(shared)
    n3 = len({frozenset((i, j)) for i, j, _ in edges3d})
    n2 = len({frozenset((i, j)) for i, j, _ in edges2d})
    n_union = len(union)
    overlap_fraction = n_shared / n3 if n3 else np.nan
    jaccard = n_shared / n_union if n_union else np.nan

    # ---- Reassignment diagnostics ----
    med_2Donly = np.median(d2_only2) if d2_only2.size else np.nan
    med_proj3Donly = np.median(d2_proj_only3) if d2_proj_only3.size else np.nan
    frac_2Donly_lt_proj3Donly = np.nan
    if d2_only2.size and d2_proj_only3.size:
        frac_2Donly_lt_proj3Donly = float(np.mean(d2_only2 < med_proj3Donly))

    mean_2Donly = float(np.mean(d2_only2)) if d2_only2.size else np.nan
    mean_proj3Donly = float(np.mean(d2_proj_only3)) if d2_proj_only3.size else np.nan
    frac_2Donly_lt_mean_proj3Donly = np.nan
    if d2_only2.size and d2_proj_only3.size:
        frac_2Donly_lt_mean_proj3Donly = float(np.mean(d2_only2 < mean_proj3Donly))

    # ---- FORCED: measure the 3D edge pairs in both spaces ----
    pairs3 = {frozenset((i, j)) for (i, j, _) in edges3d}
    d2_forced = _edge_lengths_from_pairs(pts2d, pairs3)

    # Use the 3D edge weights directly if present; fall back to recomputation if needed.
    d3_forced = np.array([w for (_, _, w) in edges3d], dtype=float)
    if d2_forced.size != d3_forced.size:
        d3_forced = _edge_lengths_from_pairs(pts3d, pairs3)

    d_forced = d2_forced - d3_forced
    r_forced = np.divide(
        d2_forced,
        d3_forced,
        out=np.full_like(d2_forced, np.nan, dtype=float),
        where=d3_forced > 0,
    )

    frac_shared_below_pi4 = float(np.mean(r_shared < geom_pi_over_4)) if r_shared.size else np.nan
    frac_forced_below_pi4 = float(np.mean(r_forced < geom_pi_over_4)) if r_forced.size else np.nan

    summary = {
        "counts": {
            "n_nodes": int(n_nodes),
            "n_edges_3d": int(n3),
            "n_edges_2d": int(n2),
            "n_shared": int(n_shared),
            "n_only3": int(len(only3)),
            "n_only2": int(len(only2)),
            "overlap_fraction_3d": float(overlap_fraction) if not np.isnan(overlap_fraction) else np.nan,
            "jaccard_similarity": float(jaccard) if not np.isnan(jaccard) else np.nan,
            "fraction_identical_nodes": float(frac_identical_nodes) if not np.isnan(frac_identical_nodes) else np.nan,
        },
        "length_stats": {
            "shared_3d": _stats_1d(d3_shared),
            "shared_2d": _stats_1d(d2_shared),
            "only3_3d_native": _stats_1d(d3_only3),
            "only3_2d_projected": _stats_1d(d2_proj_only3),
            "only2_2d_native": _stats_1d(d2_only2),
            "only2_3d_backprojected": _stats_1d(d3_back_only2),
            "forced_3d_native": _stats_1d(d3_forced),
            "forced_2d_measured": _stats_1d(d2_forced),
        },
        "shared_diffs": {
            "delta_2d_minus_3d": _stats_1d(d_shared),
            "ratio_2d_over_3d": _stats_1d(r_shared),
            "fraction_ratio_below_pi_over_4": float(frac_shared_below_pi4) if not np.isnan(frac_shared_below_pi4) else np.nan,
        },
        "forced_diffs": {
            "delta_2d_minus_3d": _stats_1d(d_forced),
            "ratio_2d_over_3d": _stats_1d(r_forced),
            "fraction_ratio_below_pi_over_4": float(frac_forced_below_pi4) if not np.isnan(frac_forced_below_pi4) else np.nan,
        },
        "reassignment": {
            "median_2Donly": float(med_2Donly) if not np.isnan(med_2Donly) else np.nan,
            "median_projected_3Donly": float(med_proj3Donly) if not np.isnan(med_proj3Donly) else np.nan,
            "fraction_2Donly_below_median_projected_3Donly": float(frac_2Donly_lt_proj3Donly) if not np.isnan(frac_2Donly_lt_proj3Donly) else np.nan,
            "mean_2Donly": mean_2Donly,
            "mean_projected_3Donly": mean_proj3Donly,
            "fraction_2Donly_below_mean_projected_3Donly": float(frac_2Donly_lt_mean_proj3Donly) if not np.isnan(frac_2Donly_lt_mean_proj3Donly) else np.nan,
            "delta_means_2Donly_minus_proj3Donly": float(mean_2Donly - mean_proj3Donly)
            if (not np.isnan(mean_2Donly) and not np.isnan(mean_proj3Donly))
            else np.nan,
        },
        "arrays": {
            "d3_shared": d3_shared,
            "d2_shared": d2_shared,
            "d_shared": d_shared,
            "r_shared": r_shared,
            "d3_only3": d3_only3,
            "d2_proj_only3": d2_proj_only3,
            "d2_only2": d2_only2,
            "d3_back_only2": d3_back_only2,
            "d3_forced": d3_forced,
            "d2_forced": d2_forced,
            "d_forced": d_forced,
            "r_forced": r_forced,
        },
        "constants": {
            "geom_pi_over_4": float(geom_pi_over_4),
        },
    }

    payload = {
        "n_nodes": summary["counts"]["n_nodes"],
        "n_edges_3d": summary["counts"]["n_edges_3d"],
        "n_edges_2d": summary["counts"]["n_edges_2d"],
        "n_shared": summary["counts"]["n_shared"],
        "n_only3": summary["counts"]["n_only3"],
        "n_only2": summary["counts"]["n_only2"],
        "overlap_fraction_3d": _round_or_nan(summary["counts"]["overlap_fraction_3d"]),
        "jaccard_similarity": _round_or_nan(summary["counts"]["jaccard_similarity"]),
        "fraction_identical_nodes": _round_or_nan(summary["counts"]["fraction_identical_nodes"]),
        "shared_ratio_mean": _round_or_nan(summary["shared_diffs"]["ratio_2d_over_3d"]["mean"]),
        "shared_ratio_median": _round_or_nan(summary["shared_diffs"]["ratio_2d_over_3d"]["median"]),
        "shared_frac_ratio_below_pi4": _round_or_nan(summary["shared_diffs"]["fraction_ratio_below_pi_over_4"]),
        "forced_ratio_mean": _round_or_nan(summary["forced_diffs"]["ratio_2d_over_3d"]["mean"]),
        "forced_ratio_median": _round_or_nan(summary["forced_diffs"]["ratio_2d_over_3d"]["median"]),
        "forced_frac_ratio_below_pi4": _round_or_nan(summary["forced_diffs"]["fraction_ratio_below_pi_over_4"]),
        "median_2Donly": _round_or_nan(summary["reassignment"]["median_2Donly"]),
        "median_projected_3Donly": _round_or_nan(summary["reassignment"]["median_projected_3Donly"]),
        "frac_2Donly_below_med_proj3Donly": _round_or_nan(
            summary["reassignment"]["fraction_2Donly_below_median_projected_3Donly"]
        ),
        "mean_2Donly": _round_or_nan(summary["reassignment"]["mean_2Donly"]),
        "mean_projected_3Donly": _round_or_nan(summary["reassignment"]["mean_projected_3Donly"]),
        "frac_2Donly_below_mean_proj3Donly": _round_or_nan(
            summary["reassignment"]["fraction_2Donly_below_mean_projected_3Donly"]
        ),
        "delta_means_2Donly_minus_proj3Donly": _round_or_nan(
            summary["reassignment"]["delta_means_2Donly_minus_proj3Donly"]
        ),
        "shared_len_median_3d": _round_or_nan(summary["length_stats"]["shared_3d"]["median"]),
        "shared_len_median_2d": _round_or_nan(summary["length_stats"]["shared_2d"]["median"]),
        "forced_len_median_3d": _round_or_nan(summary["length_stats"]["forced_3d_native"]["median"]),
        "forced_len_median_2d": _round_or_nan(summary["length_stats"]["forced_2d_measured"]["median"]),
        "geom_pi_over_4": _round_or_nan(summary["constants"]["geom_pi_over_4"], nd=5),
    }

    summary["paper_text_payload"] = payload
    return summary


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def print_nn_report(
    *,
    N_POINTS: int,
    RADIUS: float,
    profile: str,
    profile_label: str,
    ALPHA_DEG: float,
    BETA_DEG: float,
    GAMMA_DEG: float,
    L3: float,
    L2: float,
    edges3d: list[tuple[int, int, float]],
    edges2d: list[tuple[int, int, float]],
    stats3: dict,
    stats2: dict,
    comp: dict,
) -> None:
    """Print a user-friendly report comparing 3D and projected-2D NN graphs."""
    print("\n" + "=" * 72)
    print("Comparison: 3D vs projected 2D")
    print("=" * 72)

    print(f"Sampling: N={N_POINTS}, R={RADIUS:g}")
    print(f"Profile:  {profile} ({profile_label})")
    print(
        "Rotation: "
        f"α={ALPHA_DEG:.1f}°, β={BETA_DEG:.1f}°, γ={GAMMA_DEG:.1f}° (extrinsic Z-Y-X)"
    )

    print("\nEdge-set overlap")
    print("-" * 72)
    print(f"Shared edges: {comp['n_shared_edges']} / {N_POINTS-1}")
    print(f"Only in 3D:  {comp['n_only_in_3d']}")
    print(f"Only in 2D:  {comp['n_only_in_2d']}")
    print(f"Overlap (3D→2D recovery): {comp['overlap_fraction_3d']:.3f}")
    print(f"Jaccard similarity:        {comp['jaccard_similarity']:.3f}")

    print("\nTotal length")
    print("-" * 72)
    print(f"3D: {L3:.5f}")
    print(f"2D: {L2:.5f}")

    ratio = stats2["mean"] / stats3["mean"]
    geom_factor = np.pi / 4
    print("\nMean edge-length ratio")
    print("-" * 72)
    print(f"<l_2D>/<l_3D> = {ratio:.3f}  (geometric expectation ≈ {geom_factor:.3f})")

    print_length_stats("3D", stats3)
    print_length_stats("2D", stats2)

    if comp["n_only_in_3d"] or comp["n_only_in_2d"]:
        print("\nExample differing edges (up to 5 each)")
        print("-" * 72)
        only3_list = list(comp["only_in_3d_edges"])[:5]
        only2_list = list(comp["only_in_2d_edges"])[:5]

        for k, e in enumerate(only3_list, 1):
            i, j = tuple(sorted(e))
            w = next(w for (ii, jj, w) in edges3d if {ii, jj} == {i, j})
            print(f"3D-only #{k}: ({i}, {j}), length={w:.5f}")

        for k, e in enumerate(only2_list, 1):
            i, j = tuple(sorted(e))
            w = next(w for (ii, jj, w) in edges2d if {ii, jj} == {i, j})
            print(f"2D-only #{k}: ({i}, {j}), length={w:.5f}")


def print_projection_summary(summary, label: str = "") -> None:
    """Print a rich, user-friendly projection summary produced by this module."""
    c = summary["counts"]
    s = summary["length_stats"]
    d = summary["shared_diffs"]
    f = summary["forced_diffs"]
    r = summary["reassignment"]

    print("\n" + "=" * 72)
    print("Projection summary" + (f": {label}" if label else ""))
    print("=" * 72)

    print(
        "Counts: "
        f"nodes={c['n_nodes']}, edges(3D)={c['n_edges_3d']}, edges(2D)={c['n_edges_2d']}"
    )
    print(
        "Overlap: "
        f"shared={c['n_shared']} (only3={c['n_only3']}, only2={c['n_only2']}), "
        f"recovery(3D→2D)={c['overlap_fraction_3d']:.3f}, Jaccard={c['jaccard_similarity']:.3f}"
    )
    print(f"Identical neighbour sets (strict): {100 * c['fraction_identical_nodes']:.2f}%")

    print("\nShared edges (present in both edge sets)")
    print("-" * 72)
    print(
        f"Ratio l_2D/l_3D: mean={d['ratio_2d_over_3d']['mean']:.3f}, "
        f"median={d['ratio_2d_over_3d']['median']:.3f}, "
        f"fraction < π/4 = {summary['shared_diffs']['fraction_ratio_below_pi_over_4']:.3f}"
    )
    print(
        f"Length medians: 3D={s['shared_3d']['median']:.5f}  |  "
        f"2D={s['shared_2d']['median']:.5f}"
    )

    print("\nFORCED (3D-edge pairs measured in both spaces)")
    print("-" * 72)
    print(
        f"Ratio l_2D/l_3D: mean={f['ratio_2d_over_3d']['mean']:.3f}, "
        f"median={f['ratio_2d_over_3d']['median']:.3f}, "
        f"fraction < π/4 = {summary['forced_diffs']['fraction_ratio_below_pi_over_4']:.3f}"
    )
    print(
        f"Length medians: 3D={s['forced_3d_native']['median']:.5f}  |  "
        f"2D={s['forced_2d_measured']['median']:.5f}"
    )

    print("\nOne-sided edges")
    print("-" * 72)
    print(
        "Only-3D: "
        f"median(native 3D)={s['only3_3d_native']['median']:.5f}  |  "
        f"median(projected 2D)={s['only3_2d_projected']['median']:.5f}"
    )
    print(
        "Only-2D: "
        f"median(native 2D)={s['only2_2d_native']['median']:.5f}  |  "
        f"median(back-projected 3D)={s['only2_3d_backprojected']['median']:.5f}"
    )

    if not np.isnan(r["median_2Donly"]) and not np.isnan(r["median_projected_3Donly"]):
        comp = "shorter" if r["median_2Donly"] < r["median_projected_3Donly"] else "longer"
        print("\nReassignment diagnostic (medians)")
        print("-" * 72)
        print(
            f"median(2D-only)={r['median_2Donly']:.5f} is {comp} than "
            f"median(projected 3D-only)={r['median_projected_3Donly']:.5f}"
        )
        if not np.isnan(r["fraction_2Donly_below_median_projected_3Donly"]):
            print(
                "Fraction of 2D-only edges below median(projected 3D-only): "
                f"{100 * r['fraction_2Donly_below_median_projected_3Donly']:.1f}%"
            )

    if not np.isnan(r["mean_2Donly"]) and not np.isnan(r["mean_projected_3Donly"]):
        comp_mean = "shorter" if r["mean_2Donly"] < r["mean_projected_3Donly"] else "longer"
        print("\nReassignment diagnostic (means)")
        print("-" * 72)
        print(
            f"mean(2D-only)={r['mean_2Donly']:.5f} is {comp_mean} than "
            f"mean(projected 3D-only)={r['mean_projected_3Donly']:.5f}"
        )
        if not np.isnan(r["fraction_2Donly_below_mean_projected_3Donly"]):
            print(
                "Fraction of 2D-only edges below mean(projected 3D-only): "
                f"{100 * r['fraction_2Donly_below_mean_projected_3Donly']:.1f}%"
            )
        if not np.isnan(r["delta_means_2Donly_minus_proj3Donly"]):
            print(f"Δmeans (2D-only − projected 3D-only): {r['delta_means_2Donly_minus_proj3Donly']:.5f}")

    print("\nPaper-text payload (rounded)")
    print("-" * 72)
    print(json.dumps(summary["paper_text_payload"], indent=2))
