import os
import sys
import copy
import getopt
import logging
import itertools

import numpy as np
import scripts.continous_label_generator as cg
import scripts.skewed_distribution_generator as sg

from pathlib import Path
from globals.globals import SHARED_ROOT, DEBUG


def usage():
    print(" -------------------------------------------------------------------------")
    print(" Francisco Barros (francisco.teixeira.de.barros@tecnico.ulisboa.pt\n")
    print(" Generates a simulation file that can be used as input to an HIVE simulation\n")
    print(" Typical usage: simfile_generator.py --simfile=<name>.json\n")
    print(" Display all optional flags and other important notices: main.py --help\n")
    print(" -------------------------------------------------------------------------\n")
    sys.exit(" ")


def __in_max_stages():
    max_stages = input("Enter the maximum amount of stages [100, inf) the simulation should run: ")
    while True:
        try:
            max_stages = float(max_stages)
            if max_stages > 99:
                return int(max_stages) if not float('inf') else sys.maxsize
            max_stages = input("Maximum stages input should be a number in [100, inf)... Try again: ")
        except ValueError:
            max_stages = input("Input should be an integer.. Try again: ")
            continue


def __in_number_of_nodes(msg, lower_bound=1, upper_bound=10000):
    node_count = input(msg)
    while True:
        try:
            node_count = int(node_count)
            if lower_bound < node_count < upper_bound:
                return node_count
            node_count = input("At least two nodes should be indicated. Try again with value in [2, 9999]: ")
        except ValueError:
            node_count = input("Input should be an integer... Try again: ")
            continue


def __in_min_node_uptime(msg):
    min_uptime = input(msg)
    while True:
        try:
            min_uptime = float(min_uptime)
            if 0.0 <= min_uptime <= 100.0:
                return min_uptime
            min_uptime = input("Minimum node uptime should be in [0.0, 100.0]... Try again: ")
        except ValueError:
            min_uptime = input("Input should be an integer or a float... Try again: ")
            continue


def __in_samples_skewness():
    print("Skewness should be [-100.0, 100.0]; Negative skews shift distribution mean to bigger positive values!")
    skewness = input("Enter the desired skewness for skewnorm distribution: ")
    while True:
        try:
            skewness = float(skewness)
            if -100.0 <= skewness <= 100.0:
                return skewness
            skewness = input("Skewness should be in [-100.0, 100.0]... Try again: ")
        except ValueError:
            skewness = input("Input should be an integer or a float... Try again: ")
            continue


def __in_file_name(msg):
    file_name = input(msg).strip()
    while True:
        if not file_name:
            file_name = input("A non-blank file name is expected... Try again: ")
            continue
        if not Path(os.path.join(SHARED_ROOT, file_name)).is_file():
            logging.warning(str(file_name) + " isn't inside ~/hive/app/static/shared folder.")
            print("File not found in~/hive/app/static/shared). Running the present simfile might cause bad behaviour.")
        return file_name


def __in_yes_no(msg):
    char = input(msg)
    while True:
        if char == 'y' or char == 'Y':
            return True
        elif char == 'n' or char == 'N':
            return False
        else:
            char = input("Answer should be 'y' for yes or 'n' for no... Try again: ")


def __in_adj_matrix(msg, n):
    print(msg)
    print("Example input for 3x3 matrix nodes:\n1 1 1\n1 1 0\n0 1 1")
    print("Warning: simulations aren't well behaved when adjency matrices have absorbent nodes or transient states!\n")

    while True:
        adj = []
        try:
            for _ in range(n):
                line = input("")
                row_vector = [int(i) for i in line.strip().split()]
                adj.append(row_vector)
        except ValueError:
            print("Values must be 0 or 1 integers. Rows seperated by newlines, entries seperated by space. Try again: ")
            continue

        approved = True
        if len(adj) == n:
            for i in range(0, n):
                if len(adj[i]) == n:
                    for j in range(0, n):
                        if adj[i][j] == 1 or adj[j][i] == 0:
                            continue
                        else:
                            approved = False
                            break
                else:
                    break

                if not approved:
                    break

        if approved:
            if DEBUG:
                print(np.asarray(adj))
            return adj

        print("Expected a {}x{} matrix filled with 0s or 1s. Each row in a line... Try again: ".format(n, n))


def __init_nodes_uptime_dict():
    number_of_nodes = __in_number_of_nodes("Enter the number of nodes you wish to have in the network [2, 9999]: ")
    min_uptime = __in_min_node_uptime("Enter the mininum node uptime of nodes in the network [0.0, 100.0]: ")
    skewness = __in_samples_skewness()
    print("Please wait ¯\\_(ツ)_/¯ Generation of uptimes for each node may take a while.")
    samples = sg.generate_skewed_samples(skewness=skewness).tolist()
    print("Keep calm. We are almost there...")
    nodes_uptime_dict = {}
    for label in itertools.islice(cg.yield_label(), number_of_nodes):
        uptime = abs(samples.pop())  # gets and removes last element in samples to assign it to label
        nodes_uptime_dict[label] = round(uptime, 6) if uptime > min_uptime else round(min_uptime, 6)
    samples.clear()
    return nodes_uptime_dict


def __init_file_state_labels(desired_node_count, labels):
    chosen_labels = []
    labels_copy = copy.deepcopy(labels)

    current_node_count = 0
    while current_node_count < desired_node_count:
        current_node_count += 1
        choice = np.random.choice(a=labels_copy)
        labels_copy.remove(choice)
        chosen_labels.append(choice)

    if DEBUG:
        print("Original labels:\n{}\nLeft over labels in copies:\n{}\n".format(labels, labels_copy))

    return chosen_labels


def __init_shared_dict(labels):
    shared_dict = {}

    print(
        "Any file you want to simulate persistance of must be inside the following folder: ~/hive/app/static/shared\n"
        "You may also want to keep a backup of such file in:  ~/hive/app/static/shared/shared_backups"
    )

    add_file = True
    while add_file:
        n = __in_number_of_nodes("Enter the number of nodes that should be sharing this file: ")
        file_name = __in_file_name("Insert name of the file you wish to persist (include extension if it has one): ")
        shared_dict[file_name] = {}
        shared_dict[file_name]["state_labels"] = __init_file_state_labels(n, labels)

        if __in_yes_no("Would you like to manually construct an adjency matrix?"):
            shared_dict[file_name]["adj_matrix"] = __in_adj_matrix("Insert a row major {}x{} matrix: ".format(n, n), n)
        else:
            pass

        if __in_yes_no("Would you like to manually insert a desired distribution vector?"):
            pass
        else:
            pass

        add_file = __in_yes_no("Do you want to add more files to be shared under this simulation file?")

    return shared_dict


def main(simfile_name):
    if not simfile_name:
        sys.exit("Invalid simulation file name - blank name not allowed)...")

    os.path.join(SHARED_ROOT, simfile_name)

    nodes_uptime_dict = __init_nodes_uptime_dict()
    simfile_json = {
        "max_stages": __in_max_stages(),
        "nodes_uptime": nodes_uptime_dict,
        "shared": __init_shared_dict([*nodes_uptime_dict.keys()])
    }


"""
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
                sg.plot_uptime_distribution(bin_count_, sample_count_)
            if options in ("-s", "--simfile"):
                simfile_name_ = str(args).strip()
                main(simfile_name_)
    except getopt.GetoptError:
        usage()
"""

if __name__ == "__main__":
    __in_adj_matrix("hi", 3)
