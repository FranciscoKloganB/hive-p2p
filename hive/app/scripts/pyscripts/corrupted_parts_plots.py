"""
This script collects data
"""
import getopt
import json
import os
import sys
from itertools import zip_longest
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


def usage():
    print(" -------------------------------------------------------------------------")
    print(" Francisco Barros (francisco.teixeira.de.barros@tecnico.ulisboa.pt\n")
    print(" Collect data regarding corrupted parts from a site of output files located in referenced directory\n")
    print(" Typical usage: python corrupted_parts_plots.py --meandir=mean32 --istate=a\n")
    print(" Display all optional flags and other important notices: hive_simulation.py --help\n")
    print(" -------------------------------------------------------------------------\n")
    sys.exit(" ")


def plotvalues(epoch_means, mean, terminations, plot_fname):
    plt.figure()
    plt.title("Average part corruption over 30 simulations")
    plt.xlabel("Epoch (X)")
    plt.ylabel("Avg. Number of Corrupted Parts")
    plt.xlim(0, 720)
    plt.ylim(0, 6)
    # Trace global mean
    plt.axhline(y=mean,  label="global average", color='c', linestyle='-')
    # Trace cumulative means
    plt.plot(epoch_means, label="cumulative average")
    # Trace terminations
    # plt.axvline(x=terminations.pop(), label="at least one simulation instance ended", color='y', linestyle='--')
    # for epoch in terminations:
    #     plt.axvline(x=epoch, color='y', linestyle='--')
    # Display legends
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(plot_fname, prop=FontProperties().set_size('small'))


def process_file(filepath, avg_corrupted_parts, terminated_at_acount):
    with open(filepath) as instance:
        # Serialize json file
        json_obj = json.load(instance)
        # Update terminated_at_count so that epochs' means is not skewed by the'filled' zeros in zip_longest
        terminated = json_obj["terminated"]
        if terminated in terminated_at_acount:
            terminated_at_acount[terminated] += 1
        else:
            terminated_at_acount[terminated] = 1
        # Epoch data from [0, terminated) w.r.t. number of corrupted_parts of the current simulation instance
        data = json_obj["corrupted_parts"][:terminated]
        # Calculate and store the mean of current simulation instance
        avg_corrupted_parts.append(np.mean(data))
        # Calculate and store the mean at each epoch i of the current simulation instance
        temp_list = []
        for i in range(terminated):
            # Calculate the until current epoch; Since i starts at 0, we divide by i + 1
            temp_list.append(sum(data[:i]) / (i + 1))
        # Now return temp_list so it can be zipped by caller
        return temp_list


def get_epochs_means(avg_corrupted_parts_epoch, terminated_at_acount):
    breakpoints = sorted([epoch - 1 for epoch in terminated_at_acount.keys()], reverse=True)  # epoch 1 is index 0, epoch 720 is epoch 719
    last_breakpoint = breakpoints[0]
    next_breakpoint = breakpoints.pop()
    at = 0
    divisor = 30
    while at <= last_breakpoint:  # from 0 up to maximum of 719, inclusive
        if at == last_breakpoint:
            avg_corrupted_parts_epoch[at] /= divisor
            return avg_corrupted_parts_epoch
        elif at == next_breakpoint:
            divisor -= terminated_at_acount[next_breakpoint + 1]  # Subtract simulation instances who died at epoch <next_stop>, before doing the mean calculus
            avg_corrupted_parts_epoch[at] /= divisor
            next_breakpoint = breakpoints.pop()  # pop doesn't cause error because, if next stop is last stop, then while block does not execute
            at += 1
        else:
            avg_corrupted_parts_epoch[at] /= divisor
            at += 1


def main(directory, state):
    path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'static', 'outfiles')), directory, state)
    avg_corrupted_parts: List[float] = []
    avg_corrupted_parts_epoch: List[float] = [0.0] * 720
    terminated_at_acount: Dict[int, int] = {}
    for filename in os.listdir(path):
        _ = process_file(os.path.join(path, filename), avg_corrupted_parts, terminated_at_acount)
        avg_corrupted_parts_epoch = [sum(n) for n in zip_longest(avg_corrupted_parts_epoch, _, fillvalue=0)]
    # Calculate the global mean
    avg_corrupted_parts_mean = np.mean(avg_corrupted_parts)
    # Calculate the global mean at epoch i; Since we have a sum of means, at each epoch, we only need to divide each element by the number of seen instances
    avg_corrupted_parts_epoch = get_epochs_means(avg_corrupted_parts_epoch, terminated_at_acount)

    plotvalues(avg_corrupted_parts_epoch, avg_corrupted_parts_mean, [*terminated_at_acount.keys()], "{}-{}".format(meandir, istate))


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
    main(meandir, istate) if (meandir and istate) else usage()
