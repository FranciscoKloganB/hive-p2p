import getopt
import json
import os
import sys
from itertools import zip_longest
from typing import List, Dict

import numpy as np

import matplotlib.pyplot as plt
import _matplotlib_configs as cfg


def plot_values(instances_mean, instances_cs_mean, terminations_dict):
    plt.figure()

    plt.xlabel("Epoch")
    plt.ylabel("Avg. Moved Parts")

    plt.xlim(0, epochs)
    # plt.ylim(0, 1100)

    # Trace global mean
    plt.axhline(y=instances_mean, label="global mean", color='c', linestyle='-')

    # Trace cumulative means
    plt.plot(instances_cs_mean, label="global cumulative means")

    # Trace terminations
    # termination_keys = list(terminations_dict)
    # plt.axvline(x=terminations.pop(), label="at least one simulation instance ended", color='y', linestyle='--')
    # for epoch in terminations:
    #     plt.axvline(x=epoch, color='y', linestyle='--')

    # Display legends
    plt.legend(loc='lower right')

    plt.show()


def process_file(key, outfile_json, instances_means, terminations_dict):
    with open(outfile_json) as instance:
        j = json.load(instance)
        # Update terminated_at_count so that cumsum mean isn't skewed by 'fill'.
        # Important when different instances of the same simulation terminate
        # at different epoch times.
        terminated = j["terminated"]
        if terminated in terminations_dict:
            terminations_dict[terminated] += 1
        else:
            terminations_dict[terminated] = 1
        # Get the simulation instance relevant data from [0, terminated).
        # Terminated should be smaller or equal than __main__.epochs variable.
        data = j[key][:terminated]
        # Calculate and store the flat mean of the instance.
        instances_means.append(np.mean(data))
        # Calculate and store the mean at each epoch i of the instance.
        temp_list = []
        for i in range(terminated):
            # Calculate the until current epoch
            # Since i starts at 0, we divide by i + 1
            temp_list.append(sum(data[:i]) / (i + 1))
        # Now return temp_list so it can be zipped by caller
        return temp_list


def cum_sum_mean(cs_avg_list, terminations_dict):
    # Epoch 1 is index 0, epoch 720 is epoch 719.
    breakpoints = sorted(
        [epoch - 1 for epoch in terminations_dict], reverse=True)

    last_breakpoint = breakpoints[0]
    next_breakpoint = breakpoints.pop()
    at = 0
    divisor = 30
    while at <= last_breakpoint:  # from 0 up to maximum of 719, inclusive
        if at == last_breakpoint:
            cs_avg_list[at] /= divisor
            return cs_avg_list
        elif at == next_breakpoint:
            cs_avg_list[at] /= divisor
            # Subtract simulation instances who died at epoch <next_stop>
            # before calculating the mean.
            divisor -= terminations_dict[next_breakpoint + 1]
            # Pop call does not cause error, if next stop is last stop,
            # the while block will not execute.
            next_breakpoint = breakpoints.pop()
            at += 1
        else:
            cs_avg_list[at] /= divisor
            at += 1


if __name__ == "__main__":
    epochs = 0
    patterns = []
    targets = []
    short_opts = "e:p:t:"
    long_opts = ["epochs=", "patterns=", "targets="]
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
            if options in ("-t", "--targets"):
                targets = str(args).strip()
                if not targets:
                    sys.exit(f"Blank string is not a valid list of targets.")
                targets = targets.split(",")

        if epochs <= 0:
            sys.exit(f"Must specify epochs to allocate the plot's data arrays.")

        if len(targets) == 0:
            sys.exit(f"Must specify at least one json key to analyze.")

    except getopt.GetoptError:
        sys.exit("Usage: python outfile_plotter.py -f outfile.json")
    except ValueError:
        sys.exit("Execution arguments should have the following data types:\n"
                 "  --epochs -e (int)\n"
                 "  --patterns -p (comma seperated list of str)\n"
                 "  --targets -t (comma seperated list of str)\n")

    directory = os.path.abspath(
        os.path.join(os.getcwd(), '..', '..', '..', 'static', 'outfiles'))

    outfiles_view = os.listdir(directory)
    for pattern in patterns:
        outfiles_view = filter(lambda file: pattern in file, outfiles_view)

    for t in targets:
        # w.r.t. to named json field...
        # Stores flat simulations' mean values, e.g., 30 iterations' mean.
        instances_means: List[float] = []
        # Stores a simulation's cumulative mean on an epoch basis.
        instances_cs_mean: List[float] = [0.0] * epochs
        # Stores how many instances terminate at a given epoch.
        terminations_dict: Dict[str, int] = {}

        for file in outfiles_view:
            f = os.path.join(directory, file)
            result = process_file(t, f, instances_means, terminations_dict)
            zipped = zip_longest(instances_cs_mean, result, fillvalue=0)
            instances_cs_mean = [sum(n) for n in zipped]

        # Calculate the instances' global flat mean
        instances_mean = np.mean(instances_means)
        # Calculate the instances' global cumulative mean on a epoch basis.
        instances_cs_mean = cum_sum_mean(instances_cs_mean, terminations_dict)
        plot_values(instances_mean, instances_cs_mean, terminations_dict)
