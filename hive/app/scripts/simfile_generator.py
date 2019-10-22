import os
import sys
import getopt
import itertools

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import skewnorm
from globals.globals import SHARED_ROOT
from scripts.continous_label_generator import yield_label

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


def __generate_skewed_samples():
    max_uptime = 100.0
    skewness = -60.0  # Negative values are left skewed, positive values are right skewed. DON'T REMOVE (-) sign
    samples = skewnorm.rvs(a=skewness, size=599999)  # Skewnorm function
    samples = samples - min(samples)  # Shift the set so the minimum value is equal to zero
    samples = samples / max(samples)  # Standadize all the vlues between 0 and 1.
    samples = samples * max_uptime    # Multiply the standardized values by the maximum value.
    return samples


def __generate_skwed_samples_extended():
    samples = __generate_skewed_samples()
    n, bins, patches = plt.hist(samples, bins='auto', density=True)

    bin_count = len(n)

    total_density = 0.0
    for i in range(bin_count):
        total_density += n[i]

    probabilities = []
    total_probability = 0.0
    for i in range(bin_count):
        pvi = n[i] / total_density
        total_probability += pvi
        probabilities.append(pvi)

    if total_probability != 1.0:
        raise RuntimeError("Sample probabilities != 1.0 - Please report error at francisco.t.barros@tecnico.ulisboa.pt")


def __print_node_uptime_distribution(samples):
    n, bins, patches = plt.hist(samples, bins='auto', density=True)
    plt.title("Peer Node Uptime Distribution")
    plt.xlabel("uptime")
    plt.ylabel("frequency")
    plt.show()

def __init_nodes_uptime_dict():
    nodes_uptime_dict = {}
    number_of_nodes = __in_number_of_nodes()
    min_uptime = __in_min_node_uptime()
    print("Please wait... Generation of uptimes for each node may take a while.")
    samples = __generate_skewed_samples()
    node_uptime = np.random.choice(samples)
    for label in itertools.islice(yield_label(), number_of_nodes):
        uptime = np.random.choice(a=samples)
        nodes_uptime_dict[label] = node_uptime


# noinspection DuplicatedCode
if __name__ == "__main__":
    simfile_name = None
    try:
        options, args = getopt.getopt(sys.argv[1:], "uhs:", ["usage", "help", "simfile="])
        for options, args in options:
            if options in ("-u", "--usage"):
                usage()
            if options in ("-s", "--simfile"):
                simfile_name = str(args).strip()
    except getopt.GetoptError:
        usage()

    if not simfile_name:
        sys.exit("Invalid simulation file name - blank name not allowed)...")

    os.path.join(SHARED_ROOT, simfile_name)

    simfile_json = {}
    simfile_json["max_stages"] = __in_max_stages()
    simfile_json["nodes_uptime"] = __init_nodes_uptime_dict()
