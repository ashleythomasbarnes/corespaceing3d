"""Plotting utilities shared by notebooks and figures."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .rcParams import *  # noqa: F403  (project-wide Matplotlib style settings)

__all__ = [
    "make_binary_tail_cmap",
]

def make_binary_tail_cmap(
    name="inferno_binary",
    n_head=32,
    n_tail=256,
    head_name="binary",
    tail_name="inferno",
):
    """
    Create a stitched colormap by concatenating two existing Matplotlib colormaps.

    The output colormap is built by sampling a "head" colormap and a "tail"
    colormap on the interval [0, 1] and vertically stacking the resulting RGBA
    arrays. This is useful for hybrid visual encodings such as a short
    high-contrast prelude (e.g. ``binary``) followed by a perceptually uniform
    ramp (e.g. ``inferno``).

    Parameters
    ----------
    name : str, optional
        Name assigned to the returned colormap. Default is ``"inferno_binary"``.
    n_head : int, optional
        Number of discrete samples taken from the head colormap. Default is 32.
    n_tail : int, optional
        Number of discrete samples taken from the tail colormap. Default is 256.
    head_name : str or matplotlib.colors.Colormap, optional
        The head colormap (name or Colormap instance). Default is ``"binary"``.
    tail_name : str or matplotlib.colors.Colormap, optional
        The tail colormap (name or Colormap instance). Default is ``"inferno"``.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        A new colormap named ``name`` that consists of the head segment followed
        by the tail segment.

    Notes
    -----
    * Both segments are sampled including their endpoints (0 and 1). This can
      duplicate the seam color where the two segments meet. If you want to avoid
      a duplicated boundary color, you can change the sampling to e.g.
      ``np.linspace(0, 1, n_head, endpoint=False)`` for the head segment.
    * The resulting colormap is continuous in the sense that it is returned as a
      ``LinearSegmentedColormap``, but it is defined by the discrete samples used
      in the concatenation.

    Examples
    --------
    Default (binary head, inferno tail)
    >>> cmap = make_binary_tail_cmap()

    Swap the tail colormap
    >>> cmap = make_binary_tail_cmap(tail_name="magma")

    Use a reversed head colormap
    >>> cmap = make_binary_tail_cmap(head_name="Greys_r", name="greys_inferno")
    """
    head = plt.cm.get_cmap(head_name, n_head)
    tail = plt.cm.get_cmap(tail_name, n_tail)

    colors = np.vstack((
        head(np.linspace(0, 1, n_head)),
        tail(np.linspace(0, 1, n_tail)),
    ))
    return mcolors.LinearSegmentedColormap.from_list(name, colors)
