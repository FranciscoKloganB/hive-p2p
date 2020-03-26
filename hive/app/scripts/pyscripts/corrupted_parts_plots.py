"""
This script collects data
"""
import getopt
import json
import os
import sys
from itertools import zip_longest
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def usage():
    print(" -------------------------------------------------------------------------")
    print(" Francisco Barros (francisco.teixeira.de.barros@tecnico.ulisboa.pt\n")
    print(" Collect data regarding corrupted parts from a site of output files located in referenced directory\n")
    print(" Typical usage: python corrupted_parts_plots.py --meandir=mean32 --istate=a\n")
    print(" Display all optional flags and other important notices: hive_simulation.py --help\n")
    print(" -------------------------------------------------------------------------\n")
    sys.exit(" ")


def plotvalues(epoch_means, mean):
    # epochs_mean = y
    # mean = epochs_mean

    figure, axis = plt.subplot()
    plt.title("Corrupted Parts Plot")

    x = np.arange(start=0, stop=720)
    plt.xlabel("Epoch")
    plt.xlim(0, 720)

    plt.ylabel("Corrupted Parts")
    plt.ylim(0, 1100)

    # Plot the epoch mean data
    axis.plot(x, epoch_means, marker='o')
    # Plot the global mean
    axis.plot(x, mean, label='global mean', linestyle='--')

    plt.show()
    plt.savefig()


def main(directory, state):
    path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'static', 'outfiles')), directory, state)
    avg_corrupted_parts: List[float] = []
    avg_corrupted_parts_epoch: List[float] = [0.0] * 720
    for file in os.listdir(path):
        # Process each outfile representing one simulation instance, with similar initial conditions
        with open(os.path.join(path, file)) as instance:
            # Serialize json file
            json_obj = json.load(instance)
            # Epoch data from [0, terminated) w.r.t. number of corrupted_parts of the current simulation instance
            data = json_obj["corrupted_parts"][:]
            # Calculate and store the mean of current simulation instance
            avg_corrupted_parts.append(np.mean(data))
            # Calculate and store the mean at each epoch i of the current simulation instance
            temp_list = []
            for i in range(json_obj["terminated"]):
                # Calculate the until current epoch; Since i starts at 0, we divide by i + 1
                temp_list.append(sum(data[:i]) / (i+1))
            # Now sum the mean at each epoch with the existing means
            avg_corrupted_parts_epoch = [sum(n) for n in zip_longest(avg_corrupted_parts_epoch, temp_list, fillvalue=0)]
    # Calculate the global mean at epoch i; Since we have a sum of means, at each epoch, we only need to divide each element by the number of seen instances
    avg_corrupted_parts_epoch = [meansum/30 for meansum in avg_corrupted_parts_epoch]
    # Calculate the global mean
    avg_corrupted_parts_mean = np.mean(avg_corrupted_parts)
    plotvalues(epoch_means=avg_corrupted_parts_epoch, mean=avg_corrupted_parts_mean)


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
