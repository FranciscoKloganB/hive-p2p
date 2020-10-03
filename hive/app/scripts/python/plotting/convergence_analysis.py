"""
This script collects data
"""
import os
import sys
import json
import getopt

import numpy as np
import matplotlib.pyplot as plt
import _matplotlib_configs as cfg

from typing import List, Tuple

# region Old Plots (ACC 1.0 Paper) - Trashy Trash
def plotvalues(convergence_times_list, directory, state):
    print()
    # Format data sources
    time_in_convergence = []
    termination_epochs = []
    largest_window = []
    smallest_window = []
    for e in convergence_times_list:
        time_in_convergence.append(e[0])
        termination_epochs.append(e[1])
        largest_window.append(e[3])
        smallest_window.append(e[4])

    # Init figure and axis
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 7)

    # Format figure bar locations and groups
    width = 0.2  # the width of the bars
    simulation_instance_count = len(convergence_times_list)
    simulation_labels = ["S{}".format(i) for i in range(1, simulation_instance_count + 1)]  # label of each bar
    x = np.arange(simulation_instance_count)  # number of bars
    ax.bar(x - (3/2) * width, time_in_convergence, width, label='time in converrgence', color='darkslategrey')
    ax.bar(x - width / 2, termination_epochs, width, label='termination epoch', color='tan')
    ax.bar(x + width / 2, largest_window, width, label='largest convergence window', color='olivedrab')
    ax.bar(x + (3/2) * width, smallest_window, width, label='smallest convergence window', color='yellowgreen')
    # Set labels
    # ax.set_title("Convergence Analysis - {}i{}".format(directory, state))
    ax.set_xlabel("Simulation Instances")
    ax.set_ylabel("Epochs")
    # Build figure
    ax.set_xticks(x)
    ax.set_xticklabels(simulation_labels)
    plt.axhline(y=np.mean(time_in_convergence),  label="avg. time in convergence", color='darkcyan', linestyle='--')
    plt.axhline(y=np.mean(termination_epochs),  label="avg. termination epoch", color='darkkhaki', linestyle='--')
    # Format legend
    leg = ax.legend(loc='lower center', prop={'size': 9}, ncol=6, fancybox=True, shadow=True)
    # Get the bounding box of the original legend and shift its place
    bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
    bb.y0 -= 0.15  # yOffset
    bb.y1 -= 0.15  # yOffset
    leg.set_bbox_to_anchor(bb, transform=ax.transAxes)
    fig.tight_layout()
    # plt.show()
    plt.savefig("{}-{}-{}".format("convergence_sets", directory, state))


def process_file(filepath, convergence_times_list):
    time_in_convergence = 0
    with open(filepath) as instance:
        # Serialize json file
        json_obj = json.load(instance)
        terminated = json_obj["terminated"]
        largest_convergence_window = json_obj["largest_convergence_window"]
        data = json_obj["convergence_sets"]
        # Calculate how much time the cluster was in convergence
        smallest_convergence_window = terminated

        for convergence_set in data:
            time_in_convergence += (2 + len(convergence_set))
            if len(convergence_set) < smallest_convergence_window:
                smallest_convergence_window = len(convergence_set)

        if smallest_convergence_window == terminated:
            smallest_convergence_window = 0
        else:
            smallest_convergence_window += 2

        convergence_times_list.append(
            (time_in_convergence, terminated, time_in_convergence / terminated, largest_convergence_window, smallest_convergence_window)
        )


def main(directory, state):
    path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'static', 'outfiles')), directory, state)
    convergence_times_list: List[Tuple[int, int, float, int, int]] = []
    for filename in os.listdir(path):
        process_file(os.path.join(path, filename), convergence_times_list)
    # Calculate the global mean at epoch i; Since we have a sum of means, at each epoch, we only need to divide each element by the number of seen instances
    plotvalues(convergence_times_list, directory, state)
# endregion


def box_plot_instantaneous_convergence(outfiles_view: List[str]) -> None:
    pass


if __name__ == "__main__":
    # region args processing
    patterns = []

    epochs = 0
    skey = 0
    nkey = ""
    figure_name = ""

    short_opts = "p:e:s:n:f:"
    long_opts = ["patterns=", "epochs=", "size=", "name=", "figure_name="]

    try:
        options, args = getopt.getopt(sys.argv[1:], short_opts, long_opts)

        for options, args in options:
            if options in ("-p", "--patterns"):
                patterns = str(args).strip()
                if not patterns:
                    sys.exit(f"Blank pattern is not a valid pattern.")
                patterns = patterns.split(",")
            if options in ("-e", "--epochs"):
                epochs = int(str(args).strip())
            if options in ("-s", "--size"):
                skey = int(str(args).strip())
            if options in ("-n", "--name"):
                nkey = str(args).strip()
            if options in ("-f", "--figure_name"):
                figure_name = str(args).strip()

        if not (epochs > 0 and skey > 0):
            sys.exit(f"Must specify epochs (-e) and network size (-s).")

        if not (nkey != "" and figure_name != ""):
            sys.exit(f"System (-n) and Figure (-f) name must be specified "
                     f"for plot titling.")

    except ValueError:
        sys.exit("Execution arguments should have the following data types:\n"
                 "  --patterns -p (comma seperated list of str)\n"
                 "  --epochs -e (int)\n"
                 "  --size -s (int)\n"
                 "  --name -n (str)\n"
                 "  --figure_name -f (str)\n")

    directory = os.path.abspath(
        os.path.join(os.getcwd(), '..', '..', '..', 'static', 'outfiles'))
    # endregion

    outfiles_view = os.listdir(directory)
    for pattern in patterns:
        outfiles_view = list(filter(lambda f: pattern in f, outfiles_view))
        # Q2. Existem mais conjuntos de convergencia à medida que a simulação progride?
        # TODO:
        #  1. box plot instantenous convergence epochs.
        box_plot_instantaneous_convergence(outfiles_view)

        # Q3. Quanto tempo em média é preciso até observar a primeira convergencia na rede?
        # TODO:
        #  1. box plot for first convergence

        # Q4. Quantas partes são suficientes para um Swarm Guidance satisfatório? (250, 500, 750, 1000)
        # TODO:
        #  1. bar chart average time spent in instantenous convergence.
        #  Along with the charts and plots from Q5.

        # Q5. Fazendo a média dos vectores de distribuição, verifica-se uma proximidade ao vector ideal?
        # TODO:
        #  1. pie chart with % of times the desired density distribution was on average.
        #  2. box plot magnitude distance between average distribution and desired distribution
