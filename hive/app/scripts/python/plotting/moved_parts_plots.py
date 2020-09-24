import getopt
import json
import os
import sys
from itertools import zip_longest
from typing import List, Dict

import numpy as np

import matplotlib.pyplot as plt
import _matplotlib_configs as cfg


def plot_values(terminations):
    plt.figure()

    plt.xlabel("Epoch")
    plt.ylabel("Avg. Moved Parts")

    plt.xlim(0, epochs)
    plt.ylim(0, 1100)

    # Trace global mean
    plt.axhline(y=instances_mean, label="global mean", color='c', linestyle='-')

    # Trace cumulative means
    plt.plot(cs_avg_list, label="global cumulative means")

    # Trace terminations
    # plt.axvline(x=terminations.pop(), label="at least one simulation instance ended", color='y', linestyle='--')
    # for epoch in terminations:
    #     plt.axvline(x=epoch, color='y', linestyle='--')

    # Display legends
    plt.legend(loc='lower right')

    plt.show()


def process_file(outfile_json):
    with open(outfile_json) as instance:
        j = json.load(instance)
        # Update terminated_at_count so that cumsum mean isn't skewed by 'fill'.
        # Important when different instances of the same simulation terminate
        # at different epoch times.
        terminated = j["terminated"]
        if terminated in terminated_at_count:
            terminated_at_count[terminated] += 1
        else:
            terminated_at_count[terminated] = 1
        # Get the simulation instance relevant data from [0, terminated).
        # Terminated should be smaller or equal than __main__.epochs variable.
        data = j["blocks_moved"][:terminated]
        # Calculate and store the flat mean of the instance.
        sim_averages.append(np.mean(data))
        # Calculate and store the mean at each epoch i of the instance.
        temp_list = []
        for i in range(terminated):
            # Calculate the until current epoch
            # Since i starts at 0, we divide by i + 1
            temp_list.append(sum(data[:i]) / (i + 1))
        # Now return temp_list so it can be zipped by caller
        return temp_list


def get_epochs_means(avg_moved_parts_epoch, terminated_at_acount):
    breakpoints = sorted([epoch - 1 for epoch in terminated_at_acount], reverse=True)  # epoch 1 is index 0, epoch 720 is epoch 719
    last_breakpoint = breakpoints[0]
    next_breakpoint = breakpoints.pop()
    at = 0
    divisor = 30
    while at <= last_breakpoint:  # from 0 up to maximum of 719, inclusive
        if at == last_breakpoint:
            avg_moved_parts_epoch[at] /= divisor
            return avg_moved_parts_epoch
        elif at == next_breakpoint:
            avg_moved_parts_epoch[at] /= divisor
            divisor -= terminated_at_acount[next_breakpoint + 1]  # Subtract simulation instances who died at epoch <next_stop>, before doing the mean calculus
            next_breakpoint = breakpoints.pop()  # pop doesn't cause error because, if next stop is last stop, then while block does not execute
            at += 1
        else:
            avg_moved_parts_epoch[at] /= divisor
            at += 1


if __name__ == "__main__":
    epochs = 0
    patterns = []

    short_opts = "e:p:"
    long_opts = ["epochs=", "patterns="]
    try:
        options, args = getopt.getopt(sys.argv[1:], short_opts, long_opts)

        for options, args in options:
            if options in ("-e", "--epochs"):
                epochs = int(str(args).strip())
            if options in ("-p", "--patterns"):
                patterns = str(args).strip()
                if not patterns:
                    sys.exit(f"Blank pattern is not a valid pattern.")
                patterns = patterns.split(",")

        if epochs <= 0:
            sys.exit(f"Must specify epochs to allocate the plot's data arrays.")

    except getopt.GetoptError:
        sys.exit("Usage: python outfile_plotter.py -f outfile.json")
    except ValueError:
        sys.exit("Execution arguments should have the following data types:\n"
                 "  --epochs -e (int)\n"
                 "  --patterns -p (comma seperated list of str)\n")

    directory = os.path.abspath(
        os.path.join(os.getcwd(), '..', '..', '..', 'static', 'outfiles'))

    outfiles_view = os.listdir(directory)
    for pattern in patterns:
        outfiles_view = filter(lambda file: pattern in file, outfiles_view)

    # w.r.t. to named json field...
    # Stores flat simulations' mean values, e.g., 30 iterations' mean.
    sim_averages: List[float] = []
    # Stores a simulation's cumulative mean on an epoch basis.
    cs_avg_list: List[float] = [0.0] * epochs
    # Stores how many instances terminate at a given epoch.
    terminated_at_count: Dict[int, int] = {}

    for file in outfiles_view:
        filepath = os.path.join(directory, file)
        _ = process_file(filepath)
        cs_avg_list = [sum(n) for n in zip_longest(cs_avg_list, _, fillvalue=0)]

    # Calculate the instances' global flat mean
    instances_mean = np.mean(sim_averages)
    # Calculate the instances' global cumulative mean on a epoch by epoch basis.
    # Since we have a sum of means, at each epoch, we divide each element by
    # the number of visited instances, on an interval by interval basis.
    cs_avg_list = get_epochs_means(cs_avg_list, terminated_at_count)
    plot_values(list(terminated_at_count))
