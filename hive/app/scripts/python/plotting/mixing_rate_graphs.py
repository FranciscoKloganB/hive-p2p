from __future__ import annotations

import getopt
import json
import os
import sys
import matplotlib.pyplot as plt

from json import JSONDecodeError
from typing import OrderedDict, List, Any, Dict

from environment_settings import OUTFILE_ROOT

_SizeResultsDict: OrderedDict[str, List[float]]
_ResultsDict: OrderedDict[str, _SizeResultsDict]


def box_plot(json: Dict[str, Any]) -> None:
    """Creates a Box Plots that show the minimum, maximum, Q1, Q2, Q3 and IQR
    as well as outlyer mixing rate values of several markov matrix generating
    functions.

    Args:
        json: A readable json object.
    """


def __create_pie_chart__(
        size_key: str, sample_count: int, func_wins: Dict[str, int]) -> None:
    """Uses matplotlib.pyplot.pie_chart to create a pie chart.

    Args:
        size_key:
            The key representing the size of the matrices upon which the
            various functions were tested, i.e., if the matrices were of
            shape (8, 8), `size_key` should be "8".
        sample_count:
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

    fig1, ax1 = plt.subplots()

    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.axis('equal')
    plt.savefig(f"pie_chart_sk{size_key}-samples{sample_count}")


def pie_chart(json: Dict[str, Any]) -> None:
    """Creates Pies Chart that illustrate the frequency each function
    inside processed json file was selected as the fastest markov matrix
    converging to its respective steady state.

    Args:
        json: A readable json object.
    """

    for size_key, func_dict in json.items():
        # Hack to get sample count, i.e., the length of the List[float]
        # associated with each function name inside func_dict. They should
        # all be the same length. Not the cleanest solution, but works...
        sample_count = len(next(iter(func_dict.values())))
        # Init all function names with 0 wins.
        func_wins = {}.fromkeys(func_dict, 0)
        # Create a (K,V) View of the func_dict once for efficiency.
        func_dict_items = func_dict.items()
        # Iterate all samples decide wins and use results in pie chart plotting.
        for s in range(sample_count):
            best_func = ""
            smallest_mr = float('inf')
            for func_name, sample_mr in func_dict_items:
                if sample_mr < smallest_mr:
                    smallest_mr = sample_mr
                    best_func = func_name
            if best_func != "":
                func_wins[best_func] += 1
        __create_pie_chart__(size_key, sample_count, func_wins)


if __name__ == "__main__":
    file_name = ""
    try:
        short_opts = "bpf:"
        long_opts = ["boxplot", "piechart", "file="]
        options, args = getopt.getopt(sys.argv[1:], short_opts, long_opts)
        
        # Iterate all arguments first in search of -f option
        for options, args in options:
            if options in ("-f", "--file"):
                file_name = str(args).strip()

        file_path = os.path.join(
            os.path.abspath(
                os.path.join(
                    os.getcwd(), '..', '..', '..', 'static', 'outfiles')
            ), file_name)

        with open(file_path, "r") as file:
            json_obj = json.load(file)
            for options, args in options:
                if options in ("-f", "--file"):
                    file_name = str(args).strip()
                if options in ("-b", "--boxplot"):
                    box_plot(file)
                if options in ("-p", "--piechart"):
                    pie_chart(file)

    except getopt.GetoptError:
        sys.exit("Usage: python mixing_rate_sampler.py -s 1000 -f a_matrix_generator")
    except FileNotFoundError:
        sys.exit(f"File '{file_name}' does not exist in '{OUTFILE_ROOT}'.")
    except JSONDecodeError:
        sys.exit("Specified file exists, but seems to be an invalid JSON.")
    except ValueError:
        sys.exit("Execution arguments should have the following data types:\n"
                 "  --file -f (str)\n")
