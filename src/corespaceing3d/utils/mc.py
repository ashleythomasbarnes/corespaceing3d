"""Monte Carlo wrappers for the end-to-end pipeline.

This module provides lightweight helpers to run repeated realisations of the
`utils.main.main` pipeline, collecting edge-length arrays and summary metrics.

Notes
-----
- The pipeline is executed with ``verbose=False`` in Monte Carlo loops.
- Returned arrays are left as lists of NumPy arrays to avoid forcing padding.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .main import main
from .stats_mc import summarize_profile_stats


__all__ = [
    "run_main_mc",
]


def run_main_mc(
    N_RUNS: int,
    N: int,
    RADIUS: float,
    BASE_SEED: int,
    profile: str,
    profile_kwargs: dict | None,
    ALPHA_DEG: float,
    BETA_DEG: float,
    GAMMA_DEG: float,
    verbose: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[float], tuple[list[np.ndarray], list[np.ndarray]]]:
    """Run the main pipeline repeatedly and collect NN edge-length statistics.

    Parameters
    ----------
    N_RUNS : int
        Number of Monte Carlo realisations.
    N : int
        Number of points per realisation.
    RADIUS : float
        Sampling radius passed to :func:`utils.main.main`.
    BASE_SEED : int
        Base random seed; realisation ``k`` uses ``BASE_SEED + k``.
    profile : str
        Sampling profile passed to :func:`utils.main.main`.
    profile_kwargs : dict or None
        Extra keyword arguments forwarded to the sampler.
    ALPHA_DEG, BETA_DEG, GAMMA_DEG : float
        Extrinsic Euler rotation angles in degrees (Z, then Y, then X).

    Returns
    -------
    len3_runs : list of (M_k,) ndarray
        3D NN edge-length arrays for each realisation.
    len2_runs : list of (M_k,) ndarray
        2D NN edge-length arrays for each realisation.
    mu3_runs : list of float
        Mean 3D NN edge length for each realisation (NaN if empty).
    mu2_runs : list of float
        Mean 2D NN edge length for each realisation (NaN if empty).
    all_results : tuple
        Convenience tuple ``(len3_runs, len2_runs)``.
    verbose : bool, optional
        If True, print summary statistics after all runs complete.

    Notes
    -----
    This helper does not aggregate lengths across runs; callers may compute
    ensemble statistics as appropriate for their analysis.
    """
    len3_runs: list[np.ndarray] = []
    len2_runs: list[np.ndarray] = []
    mu3_runs: list[float] = []
    mu2_runs: list[float] = []

    pk: dict[str, Any] | None = profile_kwargs

    for k in range(N_RUNS):
        out = main(
            N_POINTS=N,
            RADIUS=RADIUS,
            SEED=BASE_SEED + k,
            profile=profile,
            profile_kwargs=pk,
            ALPHA_DEG=ALPHA_DEG,
            BETA_DEG=BETA_DEG,
            GAMMA_DEG=GAMMA_DEG,
            verbose=False,
        )
        l3 = np.asarray(out["len3"], dtype=float)
        l2 = np.asarray(out["len2"], dtype=float)
        len3_runs.append(l3)
        len2_runs.append(l2)
        mu3_runs.append(float(l3.mean()) if l3.size else float("nan"))
        mu2_runs.append(float(l2.mean()) if l2.size else float("nan"))

    all_results = (len3_runs, len2_runs)

    if verbose: 
        summarize_profile_stats(profile, len3_runs, len2_runs,
                            N_POINTS=N, RADIUS=RADIUS)

    return (len3_runs, len2_runs, mu3_runs, mu2_runs, all_results)
