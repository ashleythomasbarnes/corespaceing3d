"""Sampling utilities.

This module provides lightweight utilities for generating synthetic 3D point
sets with isotropic directions and a choice of spherically symmetric radial
profiles (e.g. uniform, power-law, Gaussian, exponential, Plummer, shells).
It also includes a Cartwright–Whitworth-style hierarchical fractal sampler.

Public API
----------
- sample_points_3d : Sample N points in 3D for a selected radial profile.
- euler_rotation_matrix : Build an extrinsic Z-Y-X Euler rotation matrix.

Notes
-----
All sampling routines are NumPy-only and accept an explicit `np.random.Generator`
for reproducibility.
"""

__all__ = [
    "sample_points_3d",
    "euler_rotation_matrix",
]

import numpy as np

def sample_points_3d(
    n: int,
    rng: np.random.Generator,
    profile: str = "uniform",  # 'uniform'|'gaussian'|'powerlaw'|'exponential'|'plummer'|'shell'|'fractal'
    radius: float | None = 1.0,  # outer truncation; None = unbounded (where sensible)
    **kwargs,
) -> np.ndarray:
    """
    Sample N points in 3D with various *spherically symmetric* radial profiles,
    centered at the origin. Directions are isotropic.

    Parameters
    ----------
    n : int
        Number of points.
    rng : np.random.Generator
        NumPy Generator for reproducibility.
    profile : str
        - 'uniform'    : constant ρ(r) within sphere of radius 'radius'.
        - 'gaussian'   : ρ(r) ∝ exp(-(r^2)/(2σ^2)).  kw: sigma (float).
        - 'powerlaw'   : ρ(r) ∝ r^p for 0≤r≤radius. kw: p (float, >-3).
        - 'exponential': ρ(r) ∝ exp(-r/r0).         kw: r0 (float>0).
        - 'plummer'    : ρ(r) ∝ (1 + r^2/a^2)^(-5/2). kw: a (float>0).
        - 'shell'      : thin/thick shell centered at r0 with Gaussian width. kw: r0, sigma.
        - 'fractal'     : alias of 'fractal_cw' (Cartwright–Whitworth hierarchical fractal; see 'fractal_cw').
        - 'fractal_cw'  : CW04-style hierarchical fractal cluster. kw: D (0< D ≤3), n_div (int≥2), jitter (bool), jitter_sigma (float), prune_to_sphere (bool).
    radius : float | None
        Outer truncation radius. Required for 'uniform' and 'powerlaw'.
        Optional for others (applied via rejection if given).
    kwargs :
        Profile-specific parameters (see above).

    Returns
    -------
    pts : (n,3) ndarray
        Cartesian coordinates of sampled points.
    """
    prof = profile.lower()

    if prof in {"uniform", "powerlaw"} and (radius is None or radius <= 0):
        raise ValueError(f"'radius' must be positive for profile='{profile}'.")

    # ----- choose sampler -----
    if prof == "uniform":
        draw_r = lambda m: _sample_r_uniform(m, radius, rng)
        return _truncate(draw_r, radius, n, rng)

    elif prof == "powerlaw":
        p = kwargs.get("p", 0.0)
        draw_r = lambda m: _sample_r_powerlaw(m, p, radius, rng)
        return _truncate(draw_r, radius, n, rng)

    elif prof == "gaussian":
        sigma = float(kwargs.get("sigma", (radius / 3.0 if radius else 1.0)))
        # we’ll draw from Maxwell (via normals) then apply optional truncation
        draw_r = lambda m: _sample_r_gaussian(m, sigma, rng)
        return _truncate(draw_r, radius, n, rng)

    elif prof == "exponential":
        r0 = float(kwargs.get("r0", (radius / 3.0 if radius else 1.0)))
        draw_r = lambda m: _sample_r_exponential(m, r0, rng)
        return _truncate(draw_r, radius, n, rng)

    elif prof == "plummer":
        a = float(kwargs.get("a", (radius / 3.0 if radius else 1.0)))
        draw_r = lambda m: _sample_r_plummer(m, a, rng)
        return _truncate(draw_r, radius, n, rng)

    elif prof == "shell":
        r0 = float(kwargs["r0"])
        sigma = float(kwargs.get("sigma", r0 * 0.05))
        draw_r = lambda m: _sample_r_shell(m, r0, sigma, rng)
        return _truncate(draw_r, radius, n, rng)

    elif prof in {"fractal", "fractal_cw"}:
        D = float(kwargs.get("D", 2.0))
        n_div = int(kwargs.get("n_div", 2))
        jitter = bool(kwargs.get("jitter", True))
        jitter_sigma = kwargs.get("jitter_sigma", None)
        prune_to_sphere = bool(kwargs.get("prune_to_sphere", True))
        R = radius if (radius is not None and np.isfinite(radius)) else 1.0
        pts = _sample_r_fractal(
            n=n,
            rng=rng,
            D=D,
            radius=R,
            n_div=n_div,
            jitter=jitter,
            jitter_sigma=jitter_sigma,
            prune_to_sphere=prune_to_sphere,
        )
        return pts
    
    else:
        raise ValueError(f"Unknown profile '{profile}'.")


# ----- radial samplers (vectorized) -----
def _sample_r_uniform(m, radius, rng: np.random.Generator) -> np.ndarray:
    """Sample radii for a uniform-density sphere (P(r) ∝ r^2) on [0, radius]."""
    # ρ = const in sphere ⇒ P(r) ∝ r^2, r = R * U^(1/3)
    u = rng.random(m)
    return radius * u ** (1/3)

def _sample_r_powerlaw(m, p, radius, rng: np.random.Generator) -> np.ndarray:
    """Sample radii for ρ(r) ∝ r^p on [0, radius] (requires p > -3)."""
    # ρ(r) ∝ r^p on [0,R] ⇒ P(r) ∝ r^{p+2} ⇒ r = R * U^(1/(p+3))
    if p <= -3:
        raise ValueError("For 'powerlaw', require p > -3.")
    u = rng.random(m)
    return radius * u ** (1.0 / (p + 3.0))

def _sample_r_gaussian(m, sigma, rng: np.random.Generator) -> np.ndarray:
    """Sample radii for an isotropic 3D Gaussian (Maxwell distribution)."""
    # 3D Gaussian (isotropic): simply sample x,y,z ~ N(0,σ^2), then radius via rejection if truncate
    # We'll return radii from Maxwell(σ) and pair with random directions to keep API consistent.
    # Maxwell(σ): r = σ * sqrt(χ2_3) ; equivalently, norm of 3 stdnorms.
    # Efficiently: draw Normals and take norms.
    x = rng.normal(scale=sigma, size=(m, 3))
    return np.linalg.norm(x, axis=1)

def _sample_r_exponential(m, r0, rng: np.random.Generator) -> np.ndarray:
    """Sample radii for ρ(r) ∝ exp(-r/r0) (equivalent to Gamma(k=3, θ=r0))."""
    # ρ(r) ∝ exp(-r/r0) ⇒ P(r) ∝ r^2 exp(-r/r0) ⇒ r ~ Gamma(k=3, θ=r0)
    return rng.gamma(shape=3.0, scale=r0, size=m)

def _sample_r_plummer(m, a, rng: np.random.Generator) -> np.ndarray:
    """Sample radii for a Plummer profile using the analytic inverse CDF."""
    # ρ(r) ∝ (1 + r^2/a^2)^(-5/2), r ∈ [0,∞)
    # CDF M(r)/M∞ = r^3 / (r^2 + a^2)^(3/2)
    # Inverse CDF: r = a * (u^(-2/3) - 1)^(-1/2)
    u = rng.random(m)
    return a * (u ** (-2.0 / 3.0) - 1.0) ** (-0.5)

def _sample_r_shell(m, r0, sigma, rng: np.random.Generator) -> np.ndarray:
    """Sample radii for a Gaussian shell centered at r0 with width sigma (reflected at 0)."""
    # Gaussian radial shell around r0 with width sigma (on r, not ρ).
    # Ensure positivity via truncated sampling.
    r = rng.normal(loc=r0, scale=sigma, size=m)
    return np.abs(r)  # simple reflection; for narrow shells, this is fine

def _sample_r_fractal(
    n: int,
    rng: np.random.Generator,
    D: float = 2.0,
    radius: float = 1.0,
    n_div: int = 2,
    jitter: bool = True,
    jitter_sigma: float | None = None,
    prune_to_sphere: bool = True,
    max_generations: int = 20,
) -> np.ndarray:
    """
    Generate a 3D hierarchical fractal point set using the CW04 recipe.

    Parameters
    ----------
    n : int
        Number of points requested.
    rng : np.random.Generator
        RNG for reproducibility.
    D : float, optional
        Fractal dimension (0 < D ≤ 3). Typical values: 1.6–2.6.
    radius : float, optional
        Outer characteristic radius for the cluster. Output is scaled to
        fall within ~this size (strict if prune_to_sphere=True).
    n_div : int, optional
        Subdivision per axis each generation (usually 2).
    jitter : bool, optional
        If True, add small Gaussian jitter within each final cell to avoid
        grid artifacts.
    jitter_sigma : float or None, optional
        Scatter as a fraction of the *half-size* of the final cells. If None,
        uses 3/n_div.
    prune_to_sphere : bool, optional
        If True, retain only points inside the unit sphere before final scaling
        (yields approximately spherical clusters). If False, the cube is
        uniformly rescaled to fit within `radius`.
    max_generations : int, optional
        Safety cap on hierarchy depth.

    Returns
    -------
    (n, 3) ndarray
        Cartesian coordinates.
    """
    if not (0.0 < D <= 3.0):
        raise ValueError("'D' must be in (0, 3].")
    if n_div < 2:
        raise ValueError("'n_div' must be ≥ 2.")
    if radius <= 0:
        raise ValueError("'radius' must be > 0.")

    # Probability that a child cell survives at each generation
    p_keep = n_div ** (D - 3.0)

    # Represent cells by (center, half_size). Start with root cube [-1,1]^3 (half-size=1).
    cells_centers = np.array([[0.0, 0.0, 0.0]])
    half_size = 1.0

    final_centers = None
    final_h = None

    for gen in range(max_generations):
        # Subdivide every current cell into n_div^3 children
        h_child = half_size / n_div
        # Offsets from a parent center to the child centers along one axis
        # (positions at the centers of the sub-cells)
        axis_offsets = (np.arange(n_div) - (n_div - 1) / 2.0) * (2.0 * h_child)
        grid = np.stack(np.meshgrid(axis_offsets, axis_offsets, axis_offsets, indexing='ij'), axis=-1)
        grid = grid.reshape(-1, 3)  # (n_div^3, 3)

        # Broadcast-add grid to all parent centers to get child centers
        parents = cells_centers[:, None, :]  # (N_parent, 1, 3)
        children = parents + grid[None, :, :]  # (N_parent, n_div^3, 3)
        children = children.reshape(-1, 3)  # (N_parent * n_div^3, 3)

        # Stochastically keep children
        keep = rng.random(children.shape[0]) < p_keep
        kept_children = children[keep]

        # If nothing kept (can happen for small D), retry this generation once deterministically
        if kept_children.size == 0:
            # Keep one random child per parent to avoid extinction
            # (fallback; preserves overall hierarchy depth)
            idx = rng.integers(0, grid.shape[0], size=parents.shape[0])
            kept_children = (parents[:, 0, :] + grid[idx, :]).reshape(-1, 3)

        cells_centers = kept_children
        half_size = h_child

        # Heuristic stop: ensure we have a comfortable surplus inside sphere
        # Estimate expected inside-sphere fraction ~ volume ratio of inscribed sphere to cube
        # at the current scale (rough): f ≈ π/6 for entire cube; adequate as a lower bound.
        expected_inside = cells_centers.shape[0] * (np.pi / 6.0)
        if expected_inside >= n * 1.5:
            final_centers = cells_centers.copy()
            final_h = half_size
            break
    else:
        # Reached max_generations; use what we have
        final_centers = cells_centers
        final_h = half_size

    # Optional pruning to unit sphere before scaling
    if prune_to_sphere:
        r = np.linalg.norm(final_centers, axis=1)
        inside = r <= 1.0
        pts = final_centers[inside]
        if pts.shape[0] < n:
            # If too few, disable pruning as a fallback
            pts = final_centers
    else:
        pts = final_centers

    # Add small jitter within each final cell to break grid regularity
    if jitter:
        if jitter_sigma is None:
            jitter_sigma = 3 / n_div
        pts = pts + rng.normal(scale=jitter_sigma * final_h, size=pts.shape)

    # If still too few candidates, tile (rare for extreme parameters)
    if pts.shape[0] < n:
        reps = int(np.ceil(n / pts.shape[0]))
        pts = np.vstack([pts] * reps)

    # Randomly down-select to exactly n
    idx = rng.permutation(pts.shape[0])[:n]
    pts = pts[idx]

    # Scale to requested radius. If some points are outside unit sphere (when pruning disabled),
    # renormalize by the max radius to fit within `radius`.
    norms = np.linalg.norm(pts, axis=1)
    max_norm = norms.max()
    if max_norm == 0:
        scale = 0.0
    else:
        scale = radius / max_norm
    pts = pts * scale
    return pts


def _random_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw isotropically distributed unit vectors."""
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _truncate(draw_r, trunc, n, rng: np.random.Generator) -> np.ndarray:
    """Draw radii (optionally truncated) and convert to 3D Cartesian points with isotropic directions."""
    # ----- generate radii with optional truncation -----
    # Use an over-sampling loop for efficiency under truncation.
    rs = []
    need = n
    while need > 0:
        batch = max(need, int(1.3 * need))  # a bit extra to reduce loops
        r_try = draw_r(batch)
        if trunc is not None and np.isfinite(trunc):
            keep = r_try <= trunc
            r_keep = r_try[keep]
        else:
            r_keep = r_try
        if r_keep.size > 0:
            rs.append(r_keep[:need])
            need -= min(need, r_keep.size)

    r = np.concatenate(rs)
    # Guard against exact zeros causing direction issues (rare)
    eps = np.finfo(float).tiny
    r = np.maximum(r, eps)

    # ----- assign isotropic directions -----
    dirs = _random_unit_vectors(n, rng)
    pts = dirs * r[:, None]

    return pts


def euler_rotation_matrix(alpha, beta, gamma, degrees=True) -> np.ndarray:
    """Return an extrinsic Z-Y-X Euler rotation matrix.

    Applies extrinsic rotations about Z (alpha), then Y (beta), then X (gamma).

    Parameters
    ----------
    alpha, beta, gamma : float
        Rotation angles.
    degrees : bool, optional
        If True, interpret angles as degrees. If False, angles are radians.

    Returns
    -------
    R : (3, 3) ndarray
        Rotation matrix.
    """
    if degrees:
        alpha = np.deg2rad(alpha)
        beta  = np.deg2rad(beta)
        gamma = np.deg2rad(gamma)

    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    Rz = np.array([[ ca, -sa, 0],
                   [ sa,  ca, 0],
                   [  0,   0, 1]])
    Ry = np.array([[ cb, 0, sb],
                   [  0, 1,  0],
                   [-sb, 0, cb]])
    Rx = np.array([[ 1,  0,   0],
                   [ 0, cg, -sg],
                   [ 0, sg,  cg]])
    return Rz @ Ry @ Rx
