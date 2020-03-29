"""
This script collects data
"""
import getopt
import json
import os
import sys
from itertools import zip_longest
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


def usage():
    print(" -------------------------------------------------------------------------")
    print(" Francisco Barros (francisco.teixeira.de.barros@tecnico.ulisboa.pt\n")
    print(" Collect data regarding lost parts from a collection of output files located in referenced directory\n")
    print(" Typical usage: python convergence_sets_plots.py --meandir=mean32 --istate=a\n")
    print(" Display all optional flags and other important notices: hive_simulation.py --help\n")
    print(" -------------------------------------------------------------------------\n")
    sys.exit(" ")


def plotvalues(convergence_times_list, directory, state):
    time_in_convergence = convergence_times_list[0]
    termination_epochs = convergence_times_list[1]
    largest_window = convergence_times_list[3]
    smallest_window = convergence_times_list[4]

    # colors: time_in_convergence, termination_epochs, largest_window, smallest_window
    colors = ['blue', 'red', 'cyan', 'tan']
    color_labels = ["time in converrgence", "termination epoch", "largest convergence window", "smallest convergence window"]

    # TODO: Figure how to build X axis to have 30 columns and Y axis to have instance_data values
    x = []
    for i in range(len(time_in_convergence)):
        instance_data = [time_in_convergence[i], termination_epochs[i], largest_window[i], smallest_window[i]]
        x.append(instance_data)

    fig, ax = plt.subplots()
    ax.hist(x, bins=len(time_in_convergence), range=(0, 30), density=False, color=colors, label=color_labels, histtype='bar', stacked=False)
    ax.legend(loc='upper right', prop={'size': 10})

    plt.title("Convergence Analysis - iState({})".format(state))
    plt.xlabel("Simulation Instances")
    plt.ylabel("Epochs")
    plt.axhline(y=np.mean(time_in_convergence),  label="avg. time in convergence", color='green', linestyle='--')
    plt.axhline(y=np.mean(termination_epochs),  label="avg. termination epoch", color='yellow', linestyle='--')
    plt.show()
    # plt.savefig("{}-{}-{}".format("convergence_sets", directory, state))


def process_file(filepath, convergence_times_list):
    time_in_convergence = 0
    with open(filepath) as instance:
        # Serialize json file
        json_obj = json.load(instance)
        terminated = json_obj["terminated"]
        largest_convergence_window = json_obj["largest_convergence_window"]
        data = json_obj["convergence_sets"]
        # Calculate how much time the hive was in convergence
        smallest_convergence_window = 0
        for convergence_set in data:
            time_in_convergence += (2 + len(convergence_set))
            if len(convergence_set) > smallest_convergence_window:
                smallest_convergence_window = len(convergence_set)
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


if __name__ == "__main__":
    meandir = None
    istate = None
    try:
        options, args = getopt.getopt(sys.argv[1:], "ud:i:", ["usage", "meandir=", "istate="])
        for options, args in options:
            if options in ("-u", "--usage"):
                usage()
                sys.exit(0)
            elif options in ("-d", "--meandir"):
                meandir = str(args).strip()
            elif options in ("-i", "--istate"):
                istate = str(args).strip()
    except getopt.GetoptError:
        usage()
        sys.exit(0)
    if (meandir and istate):
        main(meandir, istate)
    else:
        main('mean32', 'a')
        # main('mean32', 'i')
        # main('mean32', 'u')
        # main('mean56', 'a')
        # main('mean56', 'i')
        # main('mean56', 'u')
        # main('mean78', 'a')
        # main('mean78', 'i')
        # main('mean78', 'u')
        # main('mean90', 'a')
        # main('mean90', 'i')
        # main('mean90', 'u')
