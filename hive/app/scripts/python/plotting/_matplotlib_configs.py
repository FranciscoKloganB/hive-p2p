import os
import math

import numpy as np

import matplotlib.style
import matplotlib.pyplot
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
from typing import Any, Optional

mpl.style.use('seaborn-deep')

color_palette = matplotlib.pyplot.style.library['seaborn-deep']['axes.prop_cycle'].by_key()['color']

title_pad = 20
labels_pad = 15
legends_pad = 12
ax_alpha = 0.8

fp_title = FontProperties(
    family='sans-serif',
    weight='bold',
    size='xx-large'
)

fp_supertitle = fp_title

fp_subtitle = FontProperties(
    family='sans-serif',
    size='x-large',
    weight='semibold'
)

fp_tick_labels = FontProperties(
    family='sans-serif',
    style='italic',
    weight='semibold',
    size='xx-large'
)

fp_pctick_labels = FontProperties(
    family='sans-serif',
    style='italic',
    weight='semibold',
    size='x-large'
)

fp_legend = FontProperties(
    family='sans-serif',
    style='italic',
    weight='roman',
    size='x-large'
)

outlyer_shape = {
    'marker': 'D',
    'markeredgecolor': (1, 1, 1, 0),
    'markeredgewidth': 0,
    'markerfacecolor': color_palette[-1],
    'alpha': ax_alpha - 0.2
}


def set_boxplot_color(bp: Any, color: str) -> None:
    """Changes the color of a boxplot.

    The coloring is a filler and the alpha is set to 0.8. Whiskers, means and
    caps will always be painted black.

    Args:
        bp:
            The boxplot object to be modified.
        color:
            A hexadecimal color string or RGBA color tuple to paint the
            boxplot with.
    """
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(ax_alpha)
    plt.setp(bp['whiskers'], color="#000000")
    plt.setp(bp['caps'], color="#000000")
    plt.setp(bp['medians'], color="#000000")


def prop_legend(color: str, label: str, lw: int = 10) -> None:
    """Gives a color a description by using a proxy handler.

    Args:
        color:
            The color to paint the boxplot with or None.
        label:
            A string describing the data the color represents or None. If color
            is not set this argument is ignored.
        lw:
            How big, in height, the color line will be in plt.legend().
    """

    plt.plot([], c=color, label=label, markersize=5, linewidth=lw)


def try_coloring(
        bp: Any, color: Optional[str] = None, label: Optional[str] = None
) -> int:
    """Tries to color a boxplot. If operation was successful.

    Args:
        bp:
            A boxplot object that may or may not be modified.
        color:
            The color to paint the boxplot with or None.
        label:
            A string describing the data the color represents or None. If color
            is not set this argument is ignored.

    Returns:
        ``1`` if color was applied to the boxplot otherwise ``0``.
    """
    if color is not None:
        set_boxplot_color(bp, color)
        if label is not None:
            prop_legend(color, label)
        return 1
    return 0


def switch_tr_spine_visibility(ax: Any, set_visible: bool = False) -> None:
    """Hides the top and the right most delimiters of a figures' axes."""
    ax.spines['right'].set_visible(set_visible)
    ax.spines['top'].set_visible(set_visible)


def save_figure(
        figname: str, figext: str = "png", figdir: Optional[str] = None
) -> None:
    """Saves a figure and closes all open figures.

    Args:
        figname:
            A name for the figure.
        figext:
            The extension the figure will be saved with.
        figdir:
            The root dir where the figure will be saved. If this argument is
            None the current working directory is used.
    """
    filepath = os.getcwd() if figdir is None else figdir
    filepath = os.path.join(filepath, f"{figname}.{figext}")
    plt.savefig(filepath, format=figext, bbox_inches="tight")
    plt.close('all')


# region Offsets
def get_barchart_offsets(nbars: int, bwidth: float = 1) -> np.ndarray:
    lim = math.floor(nbars / 2)
    sublocations = np.arange(-lim, lim + 1)
    if nbars % 2 == 0:
        sublocations = list(filter(lambda x: x != 0, sublocations))
        sublocations = list(map(
            lambda x: x + (0.5 if x < 0 else -0.5), sublocations))
        return np.asarray(sublocations) * bwidth
    return sublocations * bwidth


def get_boxplot_offsets(nbars: int, spacing: float = 0.4) -> np.ndarray:
    """Generates a array containing the offsets for a tick's boxplots.

    Args:
        nbars:
            The number of boxplots the tick contains.
        spacing:
            The space between the boxplots in each tick. Be aware that depending
            on the number of ticks containing boxplots, if the spacing is too
            big, boxplots from a tick may overlap with the ones from the next
            tick.

    Returns:
        An offset array.
    """
    lim = math.floor(nbars / 2)
    sublocations = np.arange(-lim, lim + 1)
    if nbars % 2 == 0:
        sublocations = np.asarray(list(filter(lambda x: x != 0, sublocations)))
    return spacing * sublocations
# endregion
