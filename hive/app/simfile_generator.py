import os
import sys
import copy
import json
import random
import getopt
import logging
import itertools

import numpy as np
import hivescripts.continous_label_generator as cg
import hivescripts.skewed_distribution_generator as sg

from pathlib import Path
from globals.globals import SHARED_ROOT, DEBUG
from utils.randoms import excluding_randrange


# region usage
def usage():
    print(" -------------------------------------------------------------------------")
    print(" Francisco Barros (francisco.teixeira.de.barros@tecnico.ulisboa.pt\n")
    print(" Generates a simulation file that can be used as input to an HIVE simulation\n")
    print(" Typical usage: simfile_generator.py --simfile=<name>.json\n")
    print(" Display all optional flags and other important notices: main.py --help\n")
    print(" -------------------------------------------------------------------------\n")
    sys.exit(" ")
# endregion


# region input consumption and checking functions
def __in_max_stages():
    """
    :return max_stages: the number of stages this simulation should run at most
    :type int
    """
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
    """
    :param msg: message to be printed to the user upon first input request
    :type str
    :param lower_bound: rejects any input equal or below it
    :type int
    :param upper_bound: rejects any input that is equal or above it
    :type int
    :return node_count: the number of nodes
    :type int
    """
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
    """
    :param msg: message to be printed to the user upon first input request
    :type str
    :return min_uptime: minimum node uptime value
    :type float
    """
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
    """
    :return skewness: the skew value
    :type float
    """
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
    """
    :param msg: message to be printed to the user upon first input request
    :type str
    :return file_name: the name of the file to be shared
    :type str
    """
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
    """
    :param msg: message to be printed to the user upon first input request
    :type str
    :return boolean: True or False
    :type bool
    """
    char = input(msg)
    while True:
        if char == 'y' or char == 'Y':
            return True
        elif char == 'n' or char == 'N':
            return False
        else:
            char = input("Answer should be 'y' for yes or 'n' for no... Try again: ")


def __in_adj_matrix(msg, size):
    """
    This method ensures the matrix is symmetric but doesn't to prevent transient state sets or absorbent nodes
    :param msg: message to be printed to the user upon first input request
    :type str
    :param size: the size of the square matrix (size * size)
    :type int
    :return adj_matrix: the adjency matrix representing the connections between a group of peers
    :type list<list<float>>
    """
    print(msg + "\nExample input for 3x3 matrix nodes:\n1 1 1\n1 1 0\n0 1 1")
    print("Warning: only symmetric matrices are accepted. Assymetric matrices may, but aren't guaranteed to converge!")
    print("Warning: this algorithm isn't well when adjency matrices have absorbent nodes or transient state sets!\n")

    goto_while = False
    while True:
        adj_matrix = []
        for _ in range(size):
            line = input().strip().split()
            if len(line) == size:
                try:
                    line = [*map(lambda char: int(char), line)]  # transform line in row_vector
                    adj_matrix.append(line)
                    continue  # if all lines are successfully converted, none of the print errors will occur
                except ValueError:
                    pass
            print("Matrices are expected to be {}x{} with all entries being 0s or 1s. Try again: ".format(size, size))
            goto_while = True
            break

        if goto_while:
            goto_while = False
            continue

        for i in range(size):
            for j in range(i, size):
                if (adj_matrix[i][j] == adj_matrix[j][i]) and (adj_matrix[i][j] == 0 or adj_matrix[i][j] == 1):
                    continue
                print("Matrix was square, but is either assymetric or had entries different than 0 or 1. Try again: ")
                goto_while = True
                break
            if goto_while:
                break

        if goto_while:
            goto_while = False
            continue

        if DEBUG:
            print(np.asarray(adj_matrix))

        return adj_matrix


def __in_stochastic_vector(msg, size):
    """
    This method guarantees the vector is stochastic
    :param msg: message to be printed to the user upon first input request
    :type str
    :param size: the length of the vector
    :type int
    :return row_vector: the row_vector representing the desired distribution (steady state vector)
    :type <list<float>
    """
    print(msg + "\nExample input stochatic vector for three nodes sharing a file:\n0.35 0.15 0.5")
    while True:
        row_vector = input().strip().split()
        if len(row_vector) == size:
            try:
                row_vector = [*map(lambda char: float(char), row_vector)]  # transform line in stochastic vector
                if np.sum(row_vector) == 1:
                    return row_vector
            except ValueError:
                pass
            print("Expected size {}, entries must be floats, their summation must equal 1.0. Try again: ".format(size))
# endregion


# region init and generation functions
def __init_nodes_uptime_dict():
    """
    :return nodes_uptime_dict: a dictionary the maps peers (state labels) to their machine uptimes.
    :type dict<str, float>
    """
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
    """
    :param desired_node_count: the number of peers that will be responsible for sharing a file
    :type int
    :param labels: names of all peers in the system; the length of labels must be >= desired_node_count
    :type list<str>
    :return chosen_labels: a subset of :param labels; the peers from the system that were selected for sharing
    :type list<str>
    """

    if len(labels) < desired_node_count:
        raise RuntimeError("User requested that file is shared by more peers than the number of peers in the system")

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


def __init_adj_matrix(size):
    """
    Generates a random symmetric matrix without transient state sets or absorbeent nodes
    :param size: the size of the square matrix (size * size)
    :type int
    :return adj_matrix: the adjency matrix representing the connections between a group of peers
    :type list<list<float>>
    """
    secure_random = random.SystemRandom()
    adj_matrix = [[0] * size for _ in range(size)]
    choices = [0, 1]
    for i in range(size):
        for j in range(i, size):
            probability = secure_random.uniform(0.0, 1.0)
            edge_val = np.random.choice(a=choices, p=[probability, 1-probability])
            adj_matrix[i][j] = adj_matrix[j][i] = edge_val

    # Use guilty until proven innocent approach for both checks
    for i in range(size):
        is_absorbent_or_transient = True
        for j in range(size):
            # Ensure state i can reach and be reached by some other state j, where i != j
            if adj_matrix[i][j] == 1 and i != j:
                is_absorbent_or_transient = False
                break
        if is_absorbent_or_transient:
            # make a bidirectional connection with a random state j, where i != j
            j = None
            if i == 0:
                j = random.randrange(1, size)
            elif i == size-1:
                j = random.randrange(size - 1)
            else:
                j = excluding_randrange(0, i, i+1, size)
            adj_matrix[i][j] = adj_matrix[j][i] = 1
    return adj_matrix


def __init_stochastic_vector(size):
    """
    Generates a row vector whose entries summation is one.
    :param size: the length of the vector
    :type int
    :return stochastic_vector: the row_vector representing the desired distribution (steady state vector)
    :type <list<float>
    """
    secure_random = random.SystemRandom()
    stochastic_vector = [0.0] * size
    summation_pool = 1.0

    for i in range(size):
        if i == size - 1:
            stochastic_vector[i] = summation_pool
            return stochastic_vector
        else:
            probability = secure_random.uniform(0, summation_pool)
            stochastic_vector[i] = probability
            summation_pool -= probability


def __init_shared_dict(labels):
    """
    Creates the "shared" key of simulation file (json file)
    :param labels: names of all peers in the system; the length of labels must be >= desired_node_count
    :type list<str>
    :return shared_dict: the dictionary containing data respecting files to be shared in the system
    :type dict<dict<obj>>
    """
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
            shared_dict[file_name]["adj_matrix"] = __init_adj_matrix(size=n)

        if __in_yes_no("Would you like to manually insert a desired distribution vector?"):
            shared_dict[file_name]["ddv"] = __in_stochastic_vector("Insert a stochastic vector: ".format(n), n)
        else:
            shared_dict[file_name]["ddv"] = __init_stochastic_vector(size=n)
            pass

        add_file = __in_yes_no("Do you want to add more files to be shared under this simulation file?")

    return shared_dict
# endregion


# region actual main function
def main(simfile_name):
    """
    Creates a structured json file within the user's file system that can be used as input for an HIVE system simulation
    :param simfile_name: name to be assigned to the simulation file (json file) in the user's file system
    :type str
    """
    if not simfile_name:
        sys.exit("Invalid simulation file name - blank name not allowed)...")

    file_path = os.path.join(SHARED_ROOT, simfile_name)

    nodes_uptime_dict = __init_nodes_uptime_dict()
    simfile_json = {
        "max_stages": __in_max_stages(),
        "nodes_uptime": nodes_uptime_dict,
        "shared": __init_shared_dict([*nodes_uptime_dict.keys()])
    }

    with open(file_path, 'w') as outfile:
        json.dump(simfile_json, outfile)
    # TODO Finish this function and review shared_dict
# endregion


# region terminal comsumption function __main__
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
# endregion
