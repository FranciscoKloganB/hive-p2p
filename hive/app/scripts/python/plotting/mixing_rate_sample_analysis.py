from __future__ import annotations

import getopt
import json
import sys

from _matplotlib_configs import *
from typing import OrderedDict, List, Any, Dict, Optional, Tuple

_SizeResultsDict: OrderedDict[str, List[float]]
_ResultsDict: OrderedDict[str, _SizeResultsDict]

__MIXING_RATE_HOME__ = os.path.abspath(os.path.join(
    os.getcwd(), '..', '..', '..', 'static', 'outfiles', 'mixing_rate_samples'))
__MIXING_RATE_PLOTS_HOME__ = os.path.join(__MIXING_RATE_HOME__, 'plots')


# region Helpers
def __makedirs__():
    if not os.path.exists(__MIXING_RATE_PLOTS_HOME__):
        os.mkdir(__MIXING_RATE_PLOTS_HOME__)


def __shorten_labels__(labels: List[str]) -> List[str]:
    """Shortens functions' names for better plot labeling.

    Args:
        labels:
            A collection of labels to be shortened.
    """
    blacklist = {"new_", "_transition_matrix"}
    labels_count = len(labels)
    for i in range(labels_count):
        text = labels[i]
        for word in blacklist:
            text = text.replace(word, "")
        labels[i] = text
    return labels


def __create_grouped_boxplot__(datasets: List[List[Any]],
                               dcolors: List[Optional[str]],
                               dlabels: List[Optional[str]],
                               xticks_labels: List[str],
                               xlabel: Optional[str] = None,
                               ylabel: Optional[str] = None,
                               showfliers: bool = True,
                               figname: str = "", figext: str = "png",
                               savefig: bool = True) -> Tuple[Any, Any]:
    """Creates a figure where each tick has one or more boxplots.

    Args:
        datasets:
            A list containing lists with the boxplot data. For example, if the
            figure is supposed to have one boxplot per tick than, datasets
            argument would look like ``[[a1, b1, c1]]``, if it is supposed to
            have two boxplots per tick than it would be something like
            ``[[a1, b1, c1], [a2, b2, c2]]`` and so on, where ``a1`` is the
            left-most boxplot of the left-most tick and ``cn`` is the right-most
            boxplot of the right-most tick. In this case both examples have
            three ticks, if a ``d`` entry existed, there would four ticks
            instead.
        dcolors:
            The colors used to paint each boxplot or a List of Nones.
        dlabels:
            The description that gives meaning to the colors.
        xticks_labels:
            A description that differentiates each tick from the next.
    """
    fig, ax = plt.subplots()

    switch_tr_spine_visibility(ax)

    colors = 0
    boxplots_per_tick = len(datasets)
    offsets = get_boxplot_offsets(boxplots_per_tick, spacing=0.8)
    for i in range(boxplots_per_tick):
        i_data = datasets[i]
        bp = plt.boxplot(i_data, whis=0.75, widths=0.7,
                         notch=True, patch_artist=True,
                         showfliers=True, flierprops=outlyer_shape,
                         positions=np.array(range(len(i_data))) * boxplots_per_tick + offsets[i])
        colors += try_coloring(bp, dcolors[i], dlabels[i])

    if colors > 0:
        plt.legend(prop=fp_legend, ncol=colors, frameon=False,
                   loc="lower center", bbox_to_anchor=(0.5, -0.5))

    xtick_count = len(xticks_labels)
    xtick_positions = range(0, xtick_count * boxplots_per_tick, 2)
    plt.xticks(xtick_positions, xticks_labels, rotation=45,
               fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")

    if xlabel is not None:
        plt.xlabel(xlabel, labelpad=labels_pad, fontproperties=fp_tick_labels)
    if ylabel is not None:
        plt.ylabel(ylabel, labelpad=labels_pad, fontproperties=fp_tick_labels)

    plt.xlim(-boxplots_per_tick, xtick_count * boxplots_per_tick)

    if savefig:
        save_figure(figname, figext, __MIXING_RATE_PLOTS_HOME__)

    return fig, ax
# endregion


# region Box Plots
def box_plot(data_dict: _ResultsDict) -> None:
    """Creates a Box Plots that show the minimum, maximum, Q1, Q2, Q3 and IQR
    as well as outlyer mixing rate values of several markov matrix generating
    functions.

    Args:
        data_dict: A readable json object.

    Example:
        A data_dict looks like the following::

            {
                "8": {
                    "go": [float, float, float],
                    "mh": [float, float, float],
                    "sdp-mh": [float, float, float]
                }
            }
    """
    # Find all possible functions in data_dict.
    func_keys = set()
    for func_slem_dict in data_dict.values():
        func_keys.update(func_slem_dict)
    # Now generate the datasets.
    # Ensure network size and function keys are always visited in same order.
    _whitelist = {'8', '16', '32'}  # , '24', '12'}
    size_keys = sorted(list(filter(lambda s: s in _whitelist, data_dict)), key=lambda s: int(s))
    func_keys = sorted(list(func_keys))

    # For each function, add it's performance for the size.
    datasets = [None] * len(func_keys)
    for i in range(len(func_keys)):
        func_data = []
        for s in size_keys:
            func_data.append(data_dict[s][func_keys[i]])
        datasets[i] = func_data

    colors = []
    required_colors = len(func_keys)
    available_colors = len(color_palette)
    colors = color_palette * math.ceil(required_colors / available_colors)
    colors = colors[:required_colors]

    __create_grouped_boxplot__(
        datasets=datasets,
        dcolors=colors,
        dlabels=func_keys,
        xticks_labels=size_keys,
        xlabel="Cluster group size",
        ylabel="SLEM",
        figname="Mixing-Rate_BP", figext="png")
# endregion


# region Pie Charts
def pie_chart(json: _ResultsDict) -> None:
    """Creates Pies Chart that illustrate the frequency each function
    inside processed json file was selected as the fastest markov matrix
    converging to its respective steady state.

    Args:
        json: A readable json object.
    """
    for size_key, func_dict in json.items():
        # Hack to get sample count, i.e., the length of the List[float]
        sample_count = len(next(iter(func_dict.values())))
        # Init all function names with 0 wins.
        func_wins = {}.fromkeys(func_dict, 0)
        # Create a (K,V) View of the func_dict once for efficiency.
        func_dict_items = func_dict.items()
        # Iterate all samples decide wins and use results in pie chart plotting.
        for i in range(sample_count):
            best_func = ""
            smallest_mr = float('inf')
            for func_name, sample_mr in func_dict_items:
                if sample_mr[i] < smallest_mr:
                    smallest_mr = sample_mr[i]
                    best_func = func_name
            if best_func != "":
                func_wins[best_func] += 1
        __create_pie_chart__(size_key, sample_count, func_wins)


def __format_pct__(pct: np.float64) -> str:
    """Formats the pie chart wedges' text

    Args:
        pct:
            The size of the wedge relative to the remaining ones. Default pie
            value is likely to be in [0.0, 100.0].
    """
    return "" if (pct < 1.0) else f"{math.floor(pct * 10 ** 2) / 10 ** 2}%"


def __create_pie_chart__(
        skey: str, slen: int, func_wins: Dict[str, int]) -> None:
    """Uses matplotlib.pyplot.pie_chart to create a pie chart.

    Args:
        skey:
            The key representing the size of the matrices upon which the
            various functions were tested, i.e., if the matrices were of
            shape (8, 8), `skey` should be "8".
        slen:
            How many times each function in `func_wins` was tested in a
            random adjacency matrix.
        func_wins:
            A collection mapping the number of times each tested function
            name was considered the theoritically fastest, i.e., the function
            that generated a markov matrix that reaches an arbitrary steady
            state faster than the remaining ones.
    """
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = __shorten_labels__([*func_wins])
    wins = np.asarray([*func_wins.values()])

    fig1, ax = plt.subplots()

    plt.title(f"Generating functions' selection frequency for networks of size {skey}",
              x=0.57, y=1,
              pad=title_pad,
              fontproperties=fp_title)

    wedges, texts, autotexts = ax.pie(
        wins,
        autopct=lambda pct: __format_pct__(pct),
        startangle=90,
        labeldistance=None,
        textprops={
            'color': 'white',
            'weight': 'bold'
        }
    )
    # bbox_to_anchor(Xanchor, Yanchor, Xc_offset,  Yc_offset)
    # axis 'equal' ensures that pie is drawn as a circle.
    leg = ax.legend(wedges,
                    labels,
                    frameon=False,
                    loc="center left",
                    bbox_to_anchor=(0.8, 0, 0, 0))
    leg.set_title("generating function",
                  prop=fp_tick_labels)
    leg._legend_box.sep = legends_pad
    ax.axis('equal')

    figname = f"{__MIXING_RATE_PLOTS_HOME__}/mr_s{skey}pc.pdf"
    plt.savefig(figname, bbox_inches="tight", format="pdf")
# endregion


if __name__ == "__main__":
    __makedirs__()
    file_name = ""
    try:
        short_opts = "f:"
        long_opts = ["file="]
        args, values = getopt.getopt(sys.argv[1:], short_opts, long_opts)
        for arg, val in args:
            if arg in ("-f", "--file"):
                file_name = str(val).strip()
    except getopt.GetoptError:
        sys.exit("Usage: python mixing_rate_sampler.py -f a_matrix_generator")
    except ValueError:
        sys.exit("Execution arguments should have the following data types:\n"
                 "  --file -f (str)\n")

    try:
        filepath = os.path.join(__MIXING_RATE_HOME__, file_name)
        with open(filepath, "r") as f:
            data = json.load(f)
            box_plot(data)
            # pie_chart(data)
    except json.JSONDecodeError:
        sys.exit("Specified file exists, but seems to be an invalid JSON.")
    except FileNotFoundError:
        sys.exit(f"File '{file_name}' does not exist in '{__MIXING_RATE_HOME__}'.")
