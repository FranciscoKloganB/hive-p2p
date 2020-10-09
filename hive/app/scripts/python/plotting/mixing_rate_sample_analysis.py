from __future__ import annotations

import getopt
import json
import math
import os
import sys

import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt
import _matplotlib_configs as cfg

from typing import OrderedDict, List, Any, Dict
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


def __set_box_color__(bp: Any, color: str) -> None:
    """Changes the colors of a boxplot.

    Args:
        bp:
            The boxplot reference object to be modified.
        color:
            A string specifying the color to apply to the boxplot in
            hexadecimal RBG.
    """
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
# endregion


# region Box Plots
def box_plot(json: _ResultsDict) -> None:
    """Creates a Box Plots that show the minimum, maximum, Q1, Q2, Q3 and IQR
    as well as outlyer mixing rate values of several markov matrix generating
    functions.

    Args:
        json: A readable json object.
    """
    for size_key, func_dict in json.items():
        # Hack to get sample count, i.e., the length of the List[float]
        sample_count = len(next(iter(func_dict.values())))
        __create_box_plot__(size_key, sample_count, func_dict)


def __create_box_plot__(
        skey: str, slen: int, func_samples: _SizeResultsDict) -> None:
    """Uses matplotlib.pyplot.pie_chart to create a pie chart.

    Args:
        skey:
            The key representing the size of the matrices upon which the
            various functions were tested, i.e., if the matrices were of
            shape (8, 8), `skey` should be "8".
        slen:
            How many times each function was sampled for matrices of size `skey`.
        func_samples:
            A collection mapping a function names to their respective mixing
            rate samples, for matrices with size `skey`.
    """
    func_count = len(func_samples)
    func_labels = __shorten_labels__([*func_samples])
    samples = [*func_samples.values()]

    outlyer_shape = {
        # 'markerfacecolor': 'g',
        'marker': 'D'
    }

    plt.figure()
    plt.boxplot(samples,
                positions=np.array(range(func_count)) * 2.0,
                flierprops=outlyer_shape,
                widths=1,
                notch=True)
    plt.xticks(ticks=range(0, func_count * 2, 2),
               labels=func_labels,
               rotation=45)

    # plt.title(f"Algorithm performance comparison for networks of size {skey}",
    #           pad=cfg.title_pad,
    #           fontproperties=cfg.fp_title)
    plt.xlabel("generating function",
               labelpad=cfg.labels_pad,
               fontproperties=cfg.fp_axis_labels)
    plt.ylabel("mixing rate",
               labelpad=cfg.labels_pad,
               fontproperties=cfg.fp_axis_labels)

    plt.xlim(-2, func_count * 2)
    plt.ylim(0.1, 1.1)

    src = pl.Path(file_name).stem
    figname = f"{__MIXING_RATE_PLOTS_HOME__}/mr_s{skey}bp.pdf"
    plt.savefig(figname, bbox_inches="tight", format="pdf")
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

    # plt.title(f"Algorithm selection frequency for networks of size {skey}",
    #           x=0.57, y=1,
    #           pad=cfg.title_pad,
    #           fontproperties=cfg.fp_title)

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
                  prop=cfg.fp_axis_labels)
    leg._legend_box.sep = cfg.legends_pad
    ax.axis('equal')

    src = pl.Path(file_name).stem
    figname = f"{__MIXING_RATE_PLOTS_HOME__}/mr_s{skey}pc.pdf"
    plt.savefig(figname, bbox_inches="tight", format="pdf")
# endregion


if __name__ == "__main__":
    __makedirs__()
    file_name = ""
    try:
        short_opts = "bpf:"
        long_opts = ["boxplot", "piechart", "file="]
        options, args = getopt.getopt(sys.argv[1:], short_opts, long_opts)
        # Iterate all arguments first in search of -f option
        for options, args in options:
            if options in ("-f", "--file"):
                file_name = str(args).strip()
        file_path = os.path.join(__MIXING_RATE_HOME__, file_name)
        file = open(file_path, "r")
        json_obj = json.load(file)
        file.close()
        # If file was succesfully read, iterate (options, args) for methods.
        options, args = getopt.getopt(sys.argv[1:], short_opts, long_opts)
        for options, args in options:
            if options in ("-b", "--boxplot"):
                box_plot(json_obj)
            if options in ("-p", "--piechart"):
                pie_chart(json_obj)
    except getopt.GetoptError:
        sys.exit(
            "Usage: python mixing_rate_sampler.py -s 1000 -f a_matrix_generator")
    except json.JSONDecodeError:
        sys.exit("Specified file exists, but seems to be an invalid JSON.")
    except ValueError:
        sys.exit("Execution arguments should have the following data types:\n"
                 "  --file -f (str)\n")
    except FileNotFoundError:
        sys.exit(
            f"File '{file_name}' does not exist in '{__MIXING_RATE_HOME__}'.")
