import os
import sys
import getopt
import itertools
import logging

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import skewnorm
from globals.globals import SHARED_ROOT
from scripts.continous_label_generator import yield_label

DEBUG = False

def usage():
    print(" -------------------------------------------------------------------------")
    print(" Francisco Barros (francisco.teixeira.de.barros@tecnico.ulisboa.pt\n")
    print(" Generates a simulation file that can be used as input to an HIVE simulation\n")
    print(" Typical usage: simfile_generator.py --simfile=<name>.json\n")
    print(" Display all optional flags and other important notices: main.py --help\n")
    print(" -------------------------------------------------------------------------\n")
    sys.exit(" ")


def __in_max_stages():
    max_stages = input("Enter the maximum amount of stages (100, inf) the simulation should run:\n")
    while True:
        try:
            max_stages = int(max_stages)
            if max_stages > 99:
                return max_stages
            print("Maximum stages input should be a number bigger or equal to 100... Try again;")
        except ValueError:
            print("Input should be an integer.. Try again;")
            continue


def __in_number_of_nodes():
    node_count = input("Enter the number of nodes you wish to have in the network (2, inf):\n")
    while True:
        try:
            node_count = int(node_count)
            if node_count > 1:
                return node_count
            print("At least two nodes should be created. Insert a number bigger or equal to 2... Try again;")
        except ValueError:
            print("Input should be an integer.. Try again;")
            continue


def __in_min_node_uptime():
    min_uptime = input("Enter the number of nodes you wish to have in the network (0.0, 100.0):\n")
    while True:
        try:
            min_uptime = float(min_uptime)
            if 0.0 <= min_uptime <= 100.0:
                return min_uptime
            print("Minimum node uptime should be between 0.0 and 100.0... Try again;")
        except ValueError:
            print("Input should be an float.. Try again;")
            continue


def __generate_skewed_samples(sample_count=10000):
    """
    If you use this sample generation, simply select pick up the elements and assign them to a label in sequence.
    # In this case sample_count is just the number of samples I wish to take. See difference w.r.t extendend version
    """
    max_uptime = 100.0
    skewness = -90.0  # Negative values are left skewed, positive values are right skewed. DON'T REMOVE (-) sign
    samples = skewnorm.rvs(a=skewness, size=sample_count)  # Skewnorm function
    samples = samples - min(samples)  # Shift the set so the minimum value is equal to zero
    samples = samples / max(samples)  # Standadize all the vlues between 0 and 1.
    samples = samples * max_uptime    # Multiply the standardized values by the maximum value.
    return samples


def __generate_skwed_samples_extended(bin_count=7001, sample_count=7001):
    """
    If you use this sample generation, use np.random.choice
    # 7001 represents all values between [30.00, 100.00] with 0.01 step
    # 800001 Could represents all values between [20.0000, 100.000] with 0.0001 step, etc
    # To define an auto number of bins use bin='auto'
    # Keep bins_count = sample_count is just an hack to facilitate np.random.choice(bins_count, sample_count)
    """
    samples = __generate_skewed_samples(sample_count)
    bin_density, bins, patches = plt.hist(samples, bins=bin_count, density=True)

    if DEBUG:
        plt.show()
        print("total_density (numpy): " + str(np.sum(bin_density)))

    size = len(bin_density)

    total_density = 0.0
    for i in range(size):
        total_density += bin_density[i]

    total_probability = 0.0
    bin_probability = bin_density
    for i in range(size):
        bin_probability[i] = bin_density[i] / total_density
        total_probability += bin_probability[i]

    if total_probability != 1.0:
        logging.warning("probability_compensation: " + str(1.0 - total_probability))

    if DEBUG:
        print("total_density (for loop): " + str(total_density))
        print("total_probability (numpy): " + str(np.sum(bin_probability)))
        print("total_probability (for loop): " + str(total_probability))
        print("number_of_bins: " + str(len(bin_density)))
        print("number_of_samples: " + str(len(samples)))

    return samples, bin_probability


def plot_uptime_distribution(bin_count, sample_count):
    samples = __generate_skewed_samples(sample_count)
    n, bins, patches = plt.hist(samples, bin_count, density=True)
    plt.title("Peer Node Uptime Distribution")
    plt.xlabel("uptime")
    plt.ylabel("density")
    plt.show()


def __init_nodes_uptime_dict():
    nodes_uptime_dict = {}
    number_of_nodes = __in_number_of_nodes()
    min_uptime = __in_min_node_uptime()
    print("Please wait... Generation of uptimes for each node may take a while.")
    samples = __generate_skewed_samples().tolist()
    for label in itertools.islice(yield_label(), number_of_nodes):
        uptime = samples.pop()  # gets and removes last element in samples to assign it to label
        nodes_uptime_dict[label] = uptime if uptime > min_uptime else min_uptime


def main(simfile_name):
    if not simfile_name:
        sys.exit("Invalid simulation file name - blank name not allowed)...")

    os.path.join(SHARED_ROOT, simfile_name)

    simfile_json = {}
    simfile_json["max_stages"] = __in_max_stages()
    simfile_json["nodes_uptime"] = __init_nodes_uptime_dict()


# noinspection DuplicatedCode
if __name__ == "__main__":
    simfile_name_ = None
    try:
        options, args = getopt.getopt(sys.argv[1:], "ups:", ["usage", "plotuptimedistr", "simfile="])
        for options, args in options:
            if options in ("-u", "--usage"):
                usage()
            if options in ("-p", "--plotuptimedistr"):
                bin_count_ = int(input("How many bins should the distribution have?"))
                sample_count_ = int(input("How many samples should be drawn?"))
                plot_uptime_distribution(bin_count_, sample_count_)
            if options in ("-s", "--simfile"):
                simfile_name_ = str(args).strip()
                main(simfile_name_)
    except getopt.GetoptError:
        usage()


