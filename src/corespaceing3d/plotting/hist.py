import numpy as np
from scipy.stats import gaussian_kde

from .rcParams import * # Import global rcParams settings


def _bins_from_data(arr, bins):
    if isinstance(bins, int):
        if arr.size == 0:
            return np.linspace(0, 1, bins + 1)
        lo, hi = arr.min(), arr.max()
        if lo == hi:
            lo, hi = ((-0.5*lo, 1.1*hi) if lo != 0 else (0.5, 1.5))
        return np.linspace(lo, hi, bins + 1)
    return np.asarray(bins)


def _hist_kde(ax, data, bins, color, label, kde=True, lw=2):
    ax.hist(data, bins=bins, density=True, alpha=0.2, edgecolor='black', color=color)
    if kde and data.size > 1:
        xg = np.linspace(bins[0], bins[-1], 600)
        ax.plot(xg, gaussian_kde(data)(xg), lw=lw, color=color, label=f"{label}")


def plot_length_histograms(ax, len3: np.ndarray, len2: np.ndarray, bins=20):
    """
    Fancy histogram + KDE overlay for 3D vs 2D MST edge lengths.
    """

    # Shared bins
    if isinstance(bins, int):
        all_lengths = np.concatenate([len3, len2])
        bins = np.linspace(all_lengths.min(), all_lengths.max(), bins + 1)

    # Plot histograms
    ax.hist(len3, bins=bins, color='tab:blue', alpha=0.4, edgecolor='black',
            density=True, label='3D')
    ax.hist(len2, bins=bins, color='tab:orange', alpha=0.4, edgecolor='black',
            density=True, label='2D')

    # KDEs
    x_grid = np.linspace(bins.min(), bins.max(), 500)
    kde3 = gaussian_kde(len3)
    kde2 = gaussian_kde(len2)
    ax.plot(x_grid, kde3(x_grid), color='tab:blue', lw=2, label='3D KDE')
    ax.plot(x_grid, kde2(x_grid), color='tab:orange', lw=2, label='2D KDE')

    # Means
    mean3, med3, sigma3 = len3.mean(), np.median(len3), np.nanstd(len3)
    mean2, med2, sigma2 = len2.mean(), np.median(len2), np.nanstd(len2)
    ax.axvline(mean3, color='tab:blue', linestyle='--', lw=1.5)
    ax.axvline(mean2, color='tab:orange', linestyle='--', lw=1.5)

    stats_text = (
        f"3D: μ={mean3:.3f}, σ={sigma3:.3f}, med={med3:.3f}\n"
        f"2D: μ={mean2:.3f}, σ={sigma2:.3f}, med={med2:.3f}"
    )
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    ax.text(0.98, 0.84, r"$\langle \ell_{\rm 3D} \rangle/ \langle \ell_{\rm 2D} \rangle$ ="f"{mean3 / mean2:.2f}",
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # Labels
    ax.set_title("Edge Length Distributions (3D vs 2D)")
    ax.set_xlabel("Edge length")
    ax.set_ylabel("Density")
    leg = ax.legend(prop={'size': 8}, facecolor='white', framealpha=1, loc="center right")
    # Now tweak the frame’s edge color
    frame = leg.get_frame()
    frame.set_edgecolor('black')
    frame.set_linewidth(0.8)
    ax.grid()

    return ax


def plot_length_histograms_shared_forced(ax_len, ax_ratio, out, len3, len2, bins=30, kde=True, bins_len=None, bins_rat=None):
    """
    Right column:
      ax_len  : histogram overlay of lengths
                - 3D NN lengths (all 3D edges)         -> gray
                - 2D lengths for SHARED edges           -> orange
                - 2D lengths for FORCED 3D pairs        -> blue
      ax_ratio: histogram overlay of ratios d2/d3
                - SHARED edges                          -> orange
                - FORCED 3D pairs                       -> blue
    """
    shared = out["shared"]
    forced = out["forced"]

    # --- panel: lengths
    # choose common binning from combined data range for fair visual comparison
    if bins_len is None:
        all_len = np.concatenate([
            forced["d3"],           # 3D lengths of 3D NN edges
            shared["d2"],           # 2D lengths (shared edges only)
            forced["d2"]            # 2D lengths (forced from all 3D edges)
        ]) if (forced["count"] or shared["count"]) else np.array([])

        bins_len = _bins_from_data(all_len, bins)

    # 3D lengths (all 3D edges)
    _hist_kde(ax_len, len3, bins_len, color='C0', label="3D (all)", kde=kde, lw=2)
    # 2D lengths (shared edges only)
    _hist_kde(ax_len, len2, bins_len, color='C1', label="2D (all)", kde=kde, lw=2)

    # if forced["count"]:
    #     _hist_kde(ax_len, forced["d3"], bins_len, color='C0',   label="3D", kde=kde)
    if shared["count"]:
        _hist_kde(ax_len, shared["d2"], bins_len, color='C1', label="2D (shared)", kde=kde)
    if forced["count"]:
        _hist_kde(ax_len, forced["d2"], bins_len, color='C2',   label="2D (forced pairs)", kde=kde)

    ax_len.set_title("Edge-length distributions")
    ax_len.set_xlabel("Length, $\ell$")
    ax_len.set_ylabel("Density")
    leg = ax_len.legend(prop={'size': 8}, frameon=True, facecolor='white'); 
    lf = leg.get_frame(); lf.set_edgecolor('black'); lf.set_linewidth(0.8)
    ax_len.grid()

    # --- panel: ratios
    if bins_rat is None:
        ratio_arrays = []
        if shared["count"]:
            ratio_arrays.append(1 / shared["ratio"])
        if forced["count"]:
            ratio_arrays.append(1 / forced["ratio"])
        all_rat = np.concatenate(ratio_arrays) if ratio_arrays else np.array([])
        bins_rat = _bins_from_data(all_rat, bins)
        
    if shared["count"]:
        _hist_kde(ax_ratio, 1/shared["ratio"], bins_rat, color='C1', label="3D/2D (shared)", kde=kde)
    if forced["count"]:
        _hist_kde(ax_ratio, 1/forced["ratio"], bins_rat, color='C2',   label="3D/2D (forced)", kde=kde)

    ax_ratio.axvline(np.pi/4, color='k', ls=':', lw=1, label='π/4')
    ax_ratio.set_title("Length ratios (3D / 2D)")
    ax_ratio.set_xlabel(r"$\ell_{\rm 3D}$ / $\ell_{\rm 2D}$")
    ax_ratio.set_ylabel("Density")
    leg = ax_ratio.legend(prop={'size': 8}, frameon=True, facecolor='white');
    lf = leg.get_frame(); lf.set_edgecolor('black'); lf.set_linewidth(0.8)
    ax_ratio.grid()
    
    return ax_len, ax_ratio
