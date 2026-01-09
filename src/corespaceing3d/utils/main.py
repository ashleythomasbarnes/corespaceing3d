"""Pipeline driver utilities.

This module provides a small end-to-end driver for the “3D vs projected 2D”
nearest-neighbour (NN) comparison pipeline:

- sample a 3D point set (various radial profiles, including a hierarchical
  fractal sampler),
- apply an Euler rotation, project to 2D, and optionally apply a simple
  friends-of-friends “beam blending” model to mimic finite angular resolution,
- compute 1-NN graphs in 3D and 2D, and
- compare the resulting edge sets and summary statistics.

Notes
-----
- The NN graph is computed via :func:`utils.compute_nn_mst.compute_nn`.
- This module is intentionally lightweight; plotting is handled elsewhere.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import distance_matrix

from .compute_nn_mst import compute_mst, compute_nn
from .print_stats import compare_msts, compare_nns, compute_nn_projection_summary, print_nn_report, print_projection_summary
from .sampling import euler_rotation_matrix, sample_points_3d
from .stats_shared_forced import _edge_sets, build_shared_arrays_full, build_forced_arrays

__all__ = [
    "apply_beam_blending",
    "main",
]


def _profile_label(profile: str, radius: float, profile_kwargs: dict | None) -> str:
    """Return a short, human-friendly label for plot titles."""
    pk = profile_kwargs or {}
    p = profile.lower()

    if p == "uniform":
        return "Uniform"
    if p == "gaussian":
        sigma = pk.get("sigma", (radius / 3 if radius else 1.0))
        return f"Gaussian, σ={sigma:.3g}"
    if p == "powerlaw":
        return f"Power-law, p={pk.get('p', 0.0):.3g}"
    if p == "exponential":
        r0 = pk.get("r0", (radius / 3 if radius else 1.0))
        return f"Exponential, r₀={r0:.3g}"
    if p == "plummer":
        a = pk.get("a", (radius / 3 if radius else 1.0))
        return f"Plummer, a={a:.3g}"
    if p == "shell":
        r0 = pk.get("r0")
        sigma = pk.get("sigma", pk.get("r0", 1) * 0.05)
        return f"Shell, r₀={r0:g}, σ={sigma:.3g}"

    # --- fractal options ---
    if p in {"fractal", "fractal_cw"}:
        # 'fractal' is an alias of the Cartwright–Whitworth hierarchical fractal
        D = pk.get("D", 2.0)
        n_div = pk.get("n_div", 2)
        jitter = pk.get("jitter", True)
        jitter_str = ", no-jitter" if (jitter is False) else ""
        return f"Fractal, D={D:.3g}, n$_\\mathrm{{div}}$={n_div}{jitter_str}"

    return profile

def _print_verbose_reports(
    *,
    N_POINTS: int,
    RADIUS: float,
    profile: str,
    profile_kwargs: dict,
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
    pts3d: np.ndarray,
    pts2d: np.ndarray,
    label: str = "",
) -> None:
    """Print the standard verbose reports used during development."""
    prof_lbl = _profile_label(profile, RADIUS, profile_kwargs)

    print_nn_report(
        N_POINTS=N_POINTS,
        RADIUS=RADIUS,
        profile=profile,
        profile_label=prof_lbl,
        ALPHA_DEG=ALPHA_DEG,
        BETA_DEG=BETA_DEG,
        GAMMA_DEG=GAMMA_DEG,
        L3=L3,
        L2=L2,
        edges3d=edges3d,
        edges2d=edges2d,
        stats3=stats3,
        stats2=stats2,
        comp=comp,
    )

    dyn = profile_kwargs.get("dyn_range", None)
    if dyn is None:
        summary = compute_nn_projection_summary(pts3d, pts2d, edges3d, edges2d)
        print_projection_summary(summary, label=label)


def apply_beam_blending(
    points_2d: np.ndarray,
    fov_radius: float,
    dyn_range: float = 50.0,
    verbose: bool = True,
) -> tuple[np.ndarray, int, float]:
    """Simulate finite 2D resolution using friends-of-friends (FoF) blending.

    Any chain of points separated by less than the effective resolution
    (beam radius) is merged into a single blended source located at the mean
    position of the group.

    Parameters
    ----------
    points_2d : (N, 2) ndarray
        Projected coordinates (after rotation).
    fov_radius : float
        Radius (or half-size) of the field of view (sets spatial scale).
    dyn_range : float, optional
        Dynamic range = FOV / resolution. Higher values correspond to finer
        resolution. The effective resolution is ``fov_radius / dyn_range``.
    verbose : bool, optional
        If True, print a short summary of blending statistics.

    Returns
    -------
    blended_points : (N', 2) ndarray
        2D coordinates after FoF blending.
    n_merged : int
        Number of original points merged away (``N - N'``).
    resolution : float
        Adopted minimum resolvable separation.

    Notes
    -----
    - Fully hierarchical friends-of-friends: if A–B < res and B–C < res, all
      {A, B, C} merge even if A–C > res.
    - Runtime is O(N^2) due to the dense pairwise distance matrix.
    """
    pts = np.asarray(points_2d, dtype=float)
    n0 = len(pts)
    res = fov_radius / dyn_range
    if n0 == 0:
        return pts, 0, res

    # Compute pairwise distances
    dmat = distance_matrix(pts, pts)

    # Build adjacency: points within resolution are linked
    adjacency = dmat < res

    # Identify connected components (FoF groups)
    n_groups, labels = connected_components(adjacency, directed=False)

    # Average positions per group (beam-merged sources)
    blended_points = np.array([pts[labels == k].mean(axis=0) for k in range(n_groups)])
    n_new = len(blended_points)
    n_merged = n0 - n_new

    if verbose:
        frac = 100 * n_merged / n0 if n0 else 0.0
        print(
            f"[Beam blending (FoF)] dyn_range={dyn_range:.1f}, res={res:.4f} → "
            f"merged {n_merged}/{n0} ({frac:.1f}%) cores into {n_new} groups"
        )

    return blended_points, n_merged, res


def main(
    N_POINTS: int = 200,
    RADIUS: float = 1.0,
    SEED: int = 42,
    profile: str = "gaussian",
    profile_kwargs: dict | None = None,
    ALPHA_DEG: float = 25.0,
    BETA_DEG: float = 10.0,
    GAMMA_DEG: float = -5.0,
    verbose: bool = True,
    method: str = "nn",
) -> dict:
    """Run the end-to-end 3D vs projected-2D NN (or MST) comparison pipeline.

    Steps
    -----
    1. Sample 3D points with the chosen radial profile.
    2. Rotate (Euler Z-Y-X, extrinsic) and project to 2D.
    3. Optionally apply FoF beam blending in the 2D plane.
    4. Compute 1-NN graphs in 3D and 2D.
    5. Compare edge sets and print summary statistics.

    Parameters
    ----------
    N_POINTS : int, optional
        Number of points to sample.
    RADIUS : float, optional
        Outer truncation radius passed to the sampler for bounded profiles.
    SEED : int, optional
        Random seed for reproducibility.
    profile : str, optional
        Radial profile name passed to :func:`utils.sampling.sample_points_3d`.
        Options: 'uniform'|'gaussian'|'powerlaw'|'exponential'|'plummer'|'shell'|'fractal'
    profile_kwargs : dict or None, optional
        Additional keyword arguments forwarded to the sampler.
    ALPHA_DEG, BETA_DEG, GAMMA_DEG : float, optional
        Extrinsic Euler rotation angles in degrees (Z, then Y, then X).
    verbose : bool, optional
        If True, print a short report.
    method: str, optional
        Method to use: 'nn' for nearest-neighbour, 'mst' for minimum spanning tree.

    Returns
    -------
    out : dict
        Dictionary containing sampled points, edge lists, statistics, and
        convenience labels for plotting.
    """
    rng = np.random.default_rng(SEED)
    pk = profile_kwargs or {}
    method_key = method.lower()
    is_mst = method_key == "mst"
    compute_edges = compute_mst if is_mst else compute_nn
    compare_edges = compare_msts if is_mst else compare_nns
    method_label = "MST" if is_mst else "NN"

    # 1) Generate 3D points
    pts3d = sample_points_3d(N_POINTS, rng, profile=profile, radius=RADIUS, **pk)

    # 2) Rotate then project to 2D (drop z)
    R = euler_rotation_matrix(ALPHA_DEG, BETA_DEG, GAMMA_DEG, degrees=True)
    pts3d_rot = pts3d @ R.T
    pts2d = pts3d_rot[:, :2].copy()

    # Apply beam blending (simulate finite 2D resolution)
    dyn = pk.get("dyn_range", None)
    if dyn is not None:
        pts2d_blend, n_blend, _res = apply_beam_blending(
            pts2d,
            fov_radius=RADIUS,
            dyn_range=dyn,
            verbose=verbose,
        )
        if verbose:
            print(f"After beam blending: {len(pts2d_blend)} points remain (merged {n_blend})")
    else:
        pts2d_blend = pts2d

    # 3) Compute edges
    L3, edges3d, _set3, len3, stats3 = compute_edges(pts3d_rot)
    L2, edges2d, _set2, len2, stats2 = compute_edges(pts2d_blend)

    # 4) Compare
    comp = compare_edges(edges3d, edges2d)

    # 5) Print report
    if verbose:
        _print_verbose_reports(
            N_POINTS=N_POINTS,
            RADIUS=RADIUS,
            profile=profile,
            profile_kwargs=pk,
            ALPHA_DEG=ALPHA_DEG,
            BETA_DEG=BETA_DEG,
            GAMMA_DEG=GAMMA_DEG,
            L3=L3,
            L2=L2,
            edges3d=edges3d,
            edges2d=edges2d,
            stats3=stats3,
            stats2=stats2,
            comp=comp,
            pts3d=pts3d_rot,
            pts2d=pts2d_blend,
            label=f"{profile} (N={N_POINTS})",
        )

    # 6) Build titles for plots
    prof_lbl = _profile_label(profile, RADIUS, pk)
    title3d = f"3D {method_label} (N={N_POINTS}, {prof_lbl})"
    title2d = f"2D {method_label} (α={ALPHA_DEG:.1f}°, β={BETA_DEG:.1f}°, γ={GAMMA_DEG:.1f}°)"

    # 7) Shared and forced stats
    shared = build_shared_arrays_full(pts3d_rot, pts2d, edges3d, edges2d)
    forced = build_forced_arrays(pts3d_rot, pts2d, edges3d)
    edge_sets = _edge_sets(edges3d, edges2d)

    return {
        "pts3d_rot": pts3d_rot,
        "pts2d": pts2d,
        "pts2d_blend": pts2d_blend,
        "edges3d": edges3d,
        "edges2d": edges2d,
        "len3": len3,
        "len2": len2,
        "stats3": stats3,
        "stats2": stats2,
        "comp": comp,
        "titles": {"title3d": title3d, "title2d": title2d},
        "config": {
            "N_POINTS": N_POINTS,
            "RADIUS": RADIUS,
            "SEED": SEED,
            "profile": profile,
            "profile_kwargs": pk,
            "ALPHA_DEG": ALPHA_DEG,
            "BETA_DEG": BETA_DEG,
            "GAMMA_DEG": GAMMA_DEG,
        },
        "forced": {
            "d2": shared["d2"],
            "d3": forced["d3"],
            "ratio": forced["ratio"],
            "count": forced["count"],
        },
        "shared": {
            "d2": shared["d2"],
            "d3": shared["d3"],
            "ratio": shared["ratio"],
            "count": shared["count"],
        },
        "edge_sets": edge_sets,
    }