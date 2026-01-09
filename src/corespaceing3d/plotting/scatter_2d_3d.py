"""2D/3D scatter plotting helpers.

This module provides small helpers to visualise a 3D point set together with an
edge list (e.g. nearest-neighbour graph / MST) and its 2D projected counterpart.

Notes
-----
- Plot aesthetics are intentionally lightweight and local to these helpers.
- Global Matplotlib rcParams are configured via the local ``rcParams`` module.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import matplotlib.lines as mlines

from .rcParams import *  # noqa: F403  (project-wide Matplotlib style settings)


__all__ = [
    "plot_3d",
    "plot_2d",
]


def plot_3d(
    ax,
    pts3d: np.ndarray,
    edges: Iterable[tuple[int, int, float]] | np.ndarray,
    title: str = "3D",
):
    """Plot a 3D point set and an edge list.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A 3D axis instance (e.g. created via ``subplot(..., projection='3d')``).
    pts3d : (N, 3) ndarray
        3D point coordinates.
    edges : iterable of (i, j, w)
        Edge list. Only the first two entries (node indices) are used; the third
        entry is accepted for compatibility with weighted edges.
    title : str, optional
        Axis title.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The same axis, modified in place.
    """
    ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], s=12, fc="C0", ec="C0", zorder=10)

    for i, j, _ in edges:
        xs = [pts3d[i, 0], pts3d[j, 0]]
        ys = [pts3d[i, 1], pts3d[j, 1]]
        zs = [pts3d[i, 2], pts3d[j, 2]]
        ax.plot(xs, ys, zs, linewidth=1.5, color="C0", zorder=0)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    return ax


def plot_2d(
    ax,
    pts2d: np.ndarray,
    edges: Iterable[tuple[int, int, float]] | np.ndarray,
    title: str = "2D (projection)",
):
    """Plot a 2D point set and an edge list.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A 2D axis instance.
    pts2d : (N, 2) ndarray
        2D point coordinates.
    edges : iterable of (i, j, w)
        Edge list. Only the first two entries (node indices) are used; the third
        entry is accepted for compatibility with weighted edges.
    title : str, optional
        Axis title.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The same axis, modified in place.
    """
    ax.scatter(pts2d[:, 0], pts2d[:, 1], s=12, fc="C1", ec="C1", zorder=12)

    for i, j, _ in edges:
        xs = [pts2d[i, 0], pts2d[j, 0]]
        ys = [pts2d[i, 1], pts2d[j, 1]]
        ax.plot(xs, ys, linewidth=1.5, color="C1", zorder=0)

    ax.set_title(title)
    ax.set_xlabel("x'")
    ax.set_ylabel("y'")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid()

    return ax


def plot_overlay_2d(
    ax,
    pts2d: np.ndarray,
    out: dict,
    show_points: bool = True,
    title: str | None = None,
):
    """
    Overlay both MSTs in the 2D projection:
      - Shared edges: solid orange
      - 2D-only edges: solid orange
      - 3D-only edges (projected): dashed blue
    """
    shared, only3, only2, _ = out["edge_sets"]

    def _plot_edges(edges, *, color, linestyle="-", linewidth=1.5):
        for e in edges:
            i, j = tuple(e)
            ax.plot(
                [pts2d[i, 0], pts2d[j, 0]],
                [pts2d[i, 1], pts2d[j, 1]],
                lw=linewidth,
                ls=linestyle,
                color=color,
            )

    _plot_edges(shared, color="tab:orange")
    _plot_edges(only2, color="tab:orange")
    _plot_edges(only3, color="tab:blue", linestyle="--")

    if show_points:
        ax.scatter(pts2d[:, 0], pts2d[:, 1], s=10, zorder=3, c="C1")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x'")
    ax.set_ylabel("y'")
    if title:
        ax.set_title(title)

    # legend
    h_only2  = mlines.Line2D([], [], color='tab:orange', lw=2, label='2D-only')
    h_only3  = mlines.Line2D([], [], color='tab:blue', lw=1.5, ls='--', label='3D-only (proj)')
    leg = ax.legend(handles=[h_only2, h_only3],
                    prop={'size': 8}, frameon=True, facecolor='white')
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0.8)
    ax.grid()
