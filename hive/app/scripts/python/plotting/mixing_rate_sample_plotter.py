from __future__ import annotations

import getopt
import json
import os
import sys
from json import JSONDecodeError
from typing import OrderedDict, List, Any, Dict

import matplotlib.pyplot as plt

_SizeResultsDict: OrderedDict[str, List[float]]
_ResultsDict: OrderedDict[str, _SizeResultsDict]

__MIXING_RATE_HOME__ = os.path.abspath(os.path.join(
    os.getcwd(), '..', '..', '..', 'static', 'outfiles', 'mixing_rate_samples'))
__MIXING_RATE_PLOTS_HOME__ = os.path.join(__MIXING_RATE_HOME__, 'plots')


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
            How many times each function in `func_wins` was tested in a
            random adjacency matrix.
        func_samples:
            A collection mapping a function names to their respective mixing
            rate samples, for matrices with size `skey`.
    """
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    fig_name = f"{__MIXING_RATE_PLOTS_HOME__}/bp_sk{skey}-samples{slen}"
    plt.savefig(fig_name, bbox_inches='tight')


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
    labels = [*func_wins.keys()]
    sizes = [*func_wins.values()]

    fig1, ax = plt.subplots()

    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')

    fig_name = f"{__MIXING_RATE_PLOTS_HOME__}/pc_sk{skey}-samples{slen}"
    plt.savefig(fig_name, bbox_inches='tight')


def __makedirs__():
    if not os.path.exists(__MIXING_RATE_PLOTS_HOME__):
        os.mkdir(__MIXING_RATE_PLOTS_HOME__)


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
    except JSONDecodeError:
        sys.exit("Specified file exists, but seems to be an invalid JSON.")
    except ValueError:
        sys.exit("Execution arguments should have the following data types:\n"
                 "  --file -f (str)\n")
    except FileNotFoundError:
        sys.exit(
            f"File '{file_name}' does not exist in '{__MIXING_RATE_HOME__}'.")
