import getopt
import json
import os
import sys
from itertools import zip_longest
from typing import List, Dict

import numpy as np

import matplotlib.pyplot as plt
import _matplotlib_configs as cfg


def plot_values(tkey, global_mean, global_cs_mean, terminations):
    termination_epochs = [int(key) for key in terminations]

    plt.figure()

    title = f"Aggregated {nkey} simulations' results on networks of size {skey}"
    plt.title(title,
              pad=cfg.title_pad,
              fontproperties=cfg.fp_title)
    plt.xlabel("epoch",
               labelpad=cfg.labels_pad,
               fontproperties=cfg.fp_axis_labels)
    plt.ylabel(f"avg. number of {tkey.replace('_', ' ')}",
               labelpad=cfg.labels_pad,
               fontproperties=cfg.fp_axis_labels)

    plt.xlim(0, max(termination_epochs))

    # Trace global mean
    plt.axhline(y=global_mean, label="global mean", color='c', linestyle='-')
    # Trace cumulative mean values
    plt.plot(global_cs_mean, label="global cumulative mean")

    # Trace terminations
    # plt.axvline(x=terminations.pop(), label="at least one simulation instance ended", color='y', linestyle='--')
    # for epoch in terminations:
    #     plt.axvline(x=epoch, color='y', linestyle='--')

    # Display legends
    plt.legend(loc='lower right')

    plt.show()


def process_file(key, outfile_json):
    with open(outfile_json) as instance:
        j = json.load(instance)
        # Update terminated_at_count so that cumsum mean isn't skewed by 'fill'.
        # Important when different instances of the same simulation terminate
        # at different epoch times.
        terminated = j["terminated"]
        if terminated in terminations:
            terminations[terminated] += 1
        else:
            terminations[terminated] = 1
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

        return temp_list


def cum_sum_mean(cs_avg_list, terminations):
    # Epoch 1 is index 0, epoch 720 is epoch 719.
    breakpoints = sorted(
        [epoch - 1 for epoch in terminations], reverse=True)

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
            divisor -= terminations[next_breakpoint + 1]
            # Pop call does not cause error, if next stop is last stop,
            # the while block will not execute.
            next_breakpoint = breakpoints.pop()
            at += 1
        else:
            cs_avg_list[at] /= divisor
            at += 1


if __name__ == "__main__":
    # region args processing
    patterns = []
    targets = []
    epochs = 0

    skey = 0
    nkey = ""

    short_opts = "p:t:e:s:n:"
    long_opts = ["patterns=", "targets=", "epochs=", "size=", "name="]

    try:
        options, args = getopt.getopt(sys.argv[1:], short_opts, long_opts)

        for options, args in options:
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
            if options in ("-e", "--epochs"):
                epochs = int(str(args).strip())
            if options in ("-s", "--size"):
                skey = int(str(args).strip())
            if options in ("-n", "--name"):
                nkey = str(args).strip()

        if not (epochs > 0 and skey > 0):
            sys.exit(f"Must specify epochs (-e) and network size (-s).")

        if len(targets) == 0:
            sys.exit(f"Must specify at least one json key (-t) to analyze.")

    except getopt.GetoptError:
        sys.exit("Usage: python outfile_plotter.py -f outfile.json")
    except ValueError:
        sys.exit("Execution arguments should have the following data types:\n"
                 "  --epochs -e (int)\n"
                 "  --size -s (int)\n"
                 "  --patterns -p (comma seperated list of str)\n"
                 "  --targets -t (comma seperated list of str)\n")

    directory = os.path.abspath(
        os.path.join(os.getcwd(), '..', '..', '..', 'static', 'outfiles'))
    # endregion

    outfiles_view = os.listdir(directory)
    for pattern in patterns:
        outfiles_view = list(filter(lambda f: pattern in f, outfiles_view))

    for t in targets:
        # w.r.t. to named json field...
        # Stores flat simulations' mean values, e.g., 30 iterations' mean.
        instances_means: List[float] = []
        # Stores a simulation's cumulative mean on an epoch basis.
        instances_cs_mean: List[float] = [0.0] * epochs
        # Stores how many instances terminate at a given epoch.
        terminations: Dict[str, int] = {}

        for file in outfiles_view:
            f = os.path.join(directory, file)
            result = process_file(t, f)
            zipped = zip_longest(instances_cs_mean, result, fillvalue=0)
            instances_cs_mean = [sum(n) for n in zipped]

        # Calculate the instances' global flat and cumulative means
        global_mean = np.mean(instances_means)
        global_cs_mean = cum_sum_mean(instances_cs_mean, terminations)

        plot_values(t, global_mean, global_cs_mean, terminations)
