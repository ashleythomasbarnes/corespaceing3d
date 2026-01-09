"""Monte Carlo summary statistics helpers."""

import numpy as np
from scipy.stats import ks_2samp, skew, kurtosis


__all__ = [
    "summarize_profile_stats",
]


def _print_header(name, label, params, N_POINTS, RADIUS):
    header = f"=== Profile: {name}"
    if label:
        header += f" — {label}"
    header += " ==="
    print(header)
    if params:
        print(f"Params: {params}")
    if N_POINTS is not None or RADIUS is not None:
        print(f"N_POINTS = {N_POINTS}, RADIUS = {RADIUS}")


def _print_ratio(label, values):
    if np.isfinite(values).any():
        mean = np.nanmean(values)
        std = np.nanstd(values, ddof=1)
        ci = np.nanpercentile(values, [16, 84])
        print(
            f"Run-level ratio ({label}) = {mean:.3f} ± {std:.3f}  "
            f"[16–84%: {ci[0]:.3f}–{ci[1]:.3f}]"
        )

def summarize_profile_stats(
    name,
    len3_runs,
    len2_runs,
    N_POINTS=None,
    RADIUS=None,
    label=None,
    params=None,
):
    """
    Print simple + clever summary stats for one profile across many runs.

    Parameters
    ----------
    name : str
        Short profile key, e.g. 'uniform', 'gaussian'.
    len3_runs, len2_runs : list of 1D arrays
        Per-run NN edge-length arrays (3D and 2D).
    N_POINTS, RADIUS : optional
        If provided, printed for context.
    label : str, optional
        Nice printable label for the profile.
    params : dict, optional
        Profile parameters to print (e.g. {'sigma':0.25}).
    """
    # Concatenate all runs
    all3 = np.concatenate(len3_runs) if len3_runs else np.array([])
    all2 = np.concatenate(len2_runs) if len2_runs else np.array([])

    # Run-level means (NaN-safe)
    mu3 = np.array([np.mean(x) if len(x) else np.nan for x in len3_runs], dtype=float)
    mu2 = np.array([np.mean(x) if len(x) else np.nan for x in len2_runs], dtype=float)

    # Ratios per run (NaN-safe)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_3D_over_2D = mu3 / mu2
        ratio_2D_over_3D = mu2 / mu3

    _print_header(name, label, params, N_POINTS, RADIUS)

    print(f"Total runs: {len(len3_runs)}")
    print(f"Total edges: 3D={all3.size}, 2D={all2.size}")

    # ---- Global (pooled) stats ----
    if all3.size:
        print(f"Global 3D mean edge length = {all3.mean():.4f} ± {all3.std(ddof=1):.4f}")
    if all2.size:
        print(f"Global 2D mean edge length = {all2.mean():.4f} ± {all2.std(ddof=1):.4f}")

    # ---- Run-level stats (variation across realizations) ----
    if np.isfinite(mu3).any():
        print(f"Run-level 3D mean = {np.nanmean(mu3):.4f} ± {np.nanstd(mu3, ddof=1):.4f}")
    if np.isfinite(mu2).any():
        print(f"Run-level 2D mean = {np.nanmean(mu2):.4f} ± {np.nanstd(mu2, ddof=1):.4f}")

    _print_ratio("3D/2D", ratio_3D_over_2D)
    _print_ratio("2D/3D", ratio_2D_over_3D)

    # ---- Medians & percentiles (pooled) ----
    if all3.size and all2.size:
        med3, med2 = np.median(all3), np.median(all2)
        print(f"Median edge lengths: 3D={med3:.4f}, 2D={med2:.4f}")
        for q in [5, 25, 50, 75, 95]:
            p3, p2 = np.percentile(all3, q), np.percentile(all2, q)
            print(f"Percentile {q:2d}: 3D={p3:.4f}, 2D={p2:.4f}")

    # ---- Distribution overlap diagnostic (range overlap) ----
    if all3.size and all2.size:
        lo_all = min(all3.min(), all2.min())
        hi_all = max(all3.max(), all2.max())
        lo_ov = max(all3.min(), all2.min())
        hi_ov = min(all3.max(), all2.max())
        ov_len = max(0.0, hi_ov - lo_ov)
        rng = max(1e-12, hi_all - lo_all)
        overlap_fraction = ov_len / rng
        print(f"Distribution overlap range fraction = {overlap_fraction:.2f}")

        # KS test
        ks_stat, ks_p = ks_2samp(all3, all2)
        print(f"Kolmogorov–Smirnov test: D={ks_stat:.3f}, p={ks_p:.3e}")

        # Coefficient of variation, skewness, kurtosis
        cv3 = all3.std(ddof=1) / all3.mean()
        cv2 = all2.std(ddof=1) / all2.mean()
        print(f"Coefficient of variation: 3D={cv3:.3f}, 2D={cv2:.3f}")
        print(f"Skewness: 3D={skew(all3):.2f}, 2D={skew(all2):.2f}")
        print(f"Kurtosis (excess): 3D={kurtosis(all3):.2f}, 2D={kurtosis(all2):.2f}")

    print()
