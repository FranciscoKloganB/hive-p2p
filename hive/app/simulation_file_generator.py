import os
import sys
import copy
import json
import getopt
import logging
import itertools

import numpy as np

from pathlib import Path
from typing import List, Dict, Any
from globals.globals import SHARED_ROOT, SIMULATION_ROOT
from scripts.pyscripts import skewed_distribution_generator as sg, normal_distribution_generator as ng, label_generator as cg


# region Input Consumption and Verification
def __in_initial_spread() -> str:
    """
    :returns str spread_mode: how files are distributed across the hive
    """
    spread_mode = input("\nSelect how files are spread across the Hives in the beggining of the simulation: "
                        "\nPress 'u' for uniform distribution, 'i' to beging closely to desired distribution, and, 'a' to give all files to one peer...\n")
    while True:
        if spread_mode in ["u", "U", "i", "I", "a", "A"]:
            return spread_mode.lower()
        spread_mode = input("Available choices are 'u', 'i', 'a'... Try again: ")


def __in_number_of_nodes(msg: str, lower_bound: int = 1, upper_bound: int = 10000) -> int:
    """
    :param str msg: message to be printed to the user upon first input request
    :param int lower_bound: rejects any input equal or below it
    :param int upper_bound: rejects any input that is equal or above it
    :return int node_count: the number of nodes
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


def __in_min_node_uptime(msg: str) -> float:
    """
    :param str msg: message to be printed to the user upon first input request
    :return float min_uptime: minimum node uptime value
    """
    min_uptime = input(msg)
    while True:
        try:
            min_uptime = float(min_uptime)
            if 0.0 <= min_uptime <= 100.0:
                return float(str(min_uptime)[:9])  # truncates valid float value to up 6 decimals w/o any rounding!
            min_uptime = input("Minimum node uptime should be in [0.0, 100.0]... Try again: ")
        except ValueError:
            min_uptime = input("Input should be an integer or a float... Try again: ")
            continue


def __in_samples_skewness() -> float:
    """
    :return float skewness: the skew value
    """
    print("\nSkewness should be [-100.0, 100.0]; Negative skews shift distribution mean to bigger positive values!")
    skewness = input("Enter the desired skewness for skewnorm distribution: ")
    while True:
        try:
            skewness = float(skewness)
            if -100.0 <= skewness < 0.0:
                return float(str(skewness)[:10])  # truncates valid float value to up 6 decimals w/o any rounding!
            elif 0.0 <= skewness <= 100.0:
                return float(str(skewness)[:9])  # truncates valid float value to up 6 decimals w/o any rounding!
            else:
                skewness = input("Skewness should be in [-100.0, 100.0]... Try again: ")
        except ValueError:
            skewness = input("Input should be an integer or a float... Try again: ")
            continue


def __in_samples_mean() -> float:
    """
    :return float mean: the mean value
    """
    print("\nMean should be [0.0, 100.0];")
    mean = input("Enter the desired mean for the normal distribution: ")
    while True:
        try:
            mean = float(mean)
            if 0.0 <= mean <= 100.0:
                return float(str(mean)[:9])  # truncates valid float value to up 6 decimals w/o any rounding!
            else:
                mean = input("Mean should be in [0, 100.0]... Try again: ")
        except ValueError:
            mean = input("Input should be an integer or a float... Try again: ")
            continue


def __in_samples_std() -> float:
    """
    :return float std: the standard deviation value
    """
    print("\nStandard deviation should be [0.0, 50.0]")
    std = input("Enter the desired standard deviation: ")
    while True:
        try:
            std = float(std)
            if 0 <= std <= 50.0:
                return float(str(std)[:9])  # truncates valid float value to up 6 decimals w/o any rounding!
            else:
                std = input("Standard deviation should be in [0.0, 50.0]... Try again: ")
        except ValueError:
            std = input("Input should be an integer or a float... Try again: ")
            continue


def __in_file_name(msg: str) -> str:
    """
    :param str msg: message to be printed to the user upon first input request
    :return str file_name: the id of the file to be shared
    """
    file_name = input(msg).strip()
    while True:
        if not file_name:
            print("Falling back to default name 'FBZ_0134.NEF' because inputed string was blank...")
            file_name = "FBZ_0134.NEF"
            # file_name = input("A non-blank file id is expected... Try again: ")
            # continue
        if not Path(os.path.join(SHARED_ROOT, file_name)).is_file():
            logging.warning(" {} isn't inside ~/hive/app/static/shared folder.".format(file_name))
            print("File not found in~/hive/app/static/shared). Running the present simfile might cause bad behaviour.")
        return file_name


def __in_yes_no(msg: str) -> bool:
    """
    :param str msg: message to be printed to the user upon first input request
    :returns bool
    """
    char = input(msg + " y/n: ").lower()
    while True:
        if char == 'y':
            return True
        elif char == 'n':
            return False
        else:
            char = input("Answer should be 'y' for yes or 'n' for no... Try again: ")


def __in_file_labels(peer_uptime_dict: Dict[str, float], peer_names: List[str]) -> List[str]:
    """
    :param Dict[str, float] peer_uptime_dict: names of all peers in the system and their uptimes;
    :param List[str] peer_names: names of all peers in the system; the length of labels must be >= desired_node_count
    :returns List[str]: a subset of :param labels_list; labels selected by the user which are known to exist
    """
    chosen_labels = input("The following labels are available:\n{}\nInsert a list of the ones you desire...\n"
                          "Example of a five label list input: a b c aa bcd\nTIP: You are not required to input"
                          "all labels manually...\nIf you assign less labels than required for the file "
                          "or accidently assign an unexisting label, the missing labels are automatically chosen"
                          "for you!\n".format(peer_uptime_dict)).strip().split(" ")
    return [*filter(lambda label: label in peer_names, chosen_labels)]
# endregion


# region Generation Functions
def __init_peer_uptime_dict() -> Dict[str, float]:
    """
    :return Dict[str, float] peers_uptime_dict: a dictionary the maps peers (state labels) to their machine uptimes.
    """
    number_of_nodes = __in_number_of_nodes("\nEnter the number of nodes you wish to have in the network [2, 9999]: ")
    min_uptime = float(str(__in_min_node_uptime("\nEnter the mininum node uptime of nodes in the network [0.0, 100.0]: "))[:9]) / 100.0

    # skewness = __in_samples_skewness()
    # samples = sg.generate_skewed_samples(skewness=skewness).tolist()
    #
    # peers_uptime_dict = {}
    # for label in itertools.islice(cg.yield_label(), number_of_nodes):
    #     uptime = abs(samples.pop()) / 100.0  # gets and removes last element in samples to assign it to label
    #     if uptime > 1.0:
    #         peers_uptime_dict[label] = 1.0
    #     elif uptime > min_uptime:
    #         peers_uptime_dict[label] = uptime
    #     else:
    #         peers_uptime_dict[label] = min_uptime  # min_uptime was already truncated in __in_min_uptime
    # samples.clear()
    # return peers_uptime_dict
    mean = __in_samples_mean()
    std = __in_samples_std()
    samples = ng.generate_samples(surveys=1, mean=mean, std=std).tolist()

    peers_uptime_dict = {}
    for label in itertools.islice(cg.yield_label(), number_of_nodes):
        uptime = abs(samples.pop()[0]) / 100.0  # gets and removes last element in samples to assign it to label
        if uptime >= 0.99:
            peers_uptime_dict[label] = 0.99
        elif min_uptime < uptime < 0.99:
            peers_uptime_dict[label] = uptime
        else:
            peers_uptime_dict[label] = min_uptime  # min_uptime was already truncated in __in_min_uptime
    samples.clear()
    return peers_uptime_dict


def __init_hive_members(desired_node_count: int, peers_uptime_dict: Dict[str, float], peer_names: List[str]) -> List[str]:
    """
    :param int desired_node_count: the number of peers that will be responsible for sharing a file
    :param Dict[str, float] peers_uptime_dict: names of all peers in the system and their uptimes;
    :param List[str] peer_names: names of all peers in the system; the length of labels must be >= desired_node_count
    :return List[str] chosen_peers: a subset of :param labels; the peers from the system that were selected for sharing
    """

    if len(peer_names) < desired_node_count:
        raise ValueError("User requested that file is shared by more peers than the number of peers in the system")

    chosen_peers = []
    peer_names_copy = copy.deepcopy(peer_names)

    if __in_yes_no("\nDo you wish to manually insert some labels for this file?"):
        chosen_peers = __in_file_labels(peers_uptime_dict, peer_names)
        peer_names_copy = [label for label in peer_names_copy if label not in chosen_peers]

    chosen_count = len(chosen_peers)
    while chosen_count < desired_node_count:
        chosen_count += 1
        choice = np.random.choice(a=peer_names_copy)
        peer_names_copy.remove(choice)
        chosen_peers.append(choice)
    return chosen_peers


def __init_shared_dict(peer_uptime_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Creates the "shared" key of simulation file (json file)
    :param Dict[str, float] peer_uptime_dict: names of all peers in the system and their uptimes
    :return Dict[Dict[Any]]shared_dict: the dictionary containing data respecting files to be shared in the system
    """
    shared_dict: Dict[str, Any] = {}
    peer_names: List[str] = [*peer_uptime_dict.keys()]

    print(
        "\nAny file you want to simulate persistance of must be inside the following folder: ~/hive/app/static/shared\n"
        "You may also want to keep a backup of such file in:  ~/hive/app/static/shared/shared_backups"
    )

    add_file: bool = True
    while add_file:
        file_name = __in_file_name("\nInsert name of the file you wish to persist (include extension if it has one): ")
        shared_dict[file_name] = {}
        shared_dict[file_name]["spread"] = __in_initial_spread()
        shared_dict[file_name]["hive_size"] = __in_number_of_nodes("Enter the number of nodes that should be sharing the next file: \n")
        # n = __in_number_of_nodes("Enter the number of nodes that should be sharing the next file: \n")
        # shared_dict[file_name]["members"] = __init_hive_members(n, peer_uptime_dict, peer_names)
        add_file = __in_yes_no("\nDo you want to add more files to be shared under this simulation file?")
    return shared_dict
# endregion


# region Main and Usage
def usage():
    print(" -------------------------------------------------------------------------")
    print(" Francisco Barros (francisco.teixeira.de.barros@tecnico.ulisboa.pt\n")
    print(" Generates a simulation file that can be used as input to an HIVE simulation\n")
    print(" Typical usage: python simulation_file_generator.py --simfile=filename.json\n")
    print(" Display all optional flags and other important notices: hive_simulation.py --help\n")
    print(" -------------------------------------------------------------------------\n")
    sys.exit(" ")


def main(simfile_name: str):
    """
    Creates a structured json file within the user's file system that can be used as input for an HIVE system simulation
    :param str simfile_name: id to be assigned to the simulation file (json file) in the user's file system
    """
    peers_uptime_dict: Dict[str, float] = __init_peer_uptime_dict()
    simfile_json: Dict[str, Any] = {
        "peers_uptime": peers_uptime_dict,
        "shared": __init_shared_dict(peers_uptime_dict)
    }

    with open(os.path.join(SIMULATION_ROOT, simfile_name), 'w') as outfile:
        json.dump(simfile_json, outfile, indent=4)
# endregion


# region terminal comsumption function __main__
# noinspection DuplicatedCode
if __name__ == "__main__":
    simfile_name_: str = ""
    try:
        options, args = getopt.getopt(sys.argv[1:], "ups:", ["usage", "plotuptimedistr", "simfile="])
        for options, args in options:
            if options in ("-u", "--usage"):
                usage()
            if options in ("-p", "--plotuptimedistr"):
                ng.plot_uptime_distribution()
            if options in ("-s", "--simfile"):
                simfile_name_ = str(args).strip()
                if simfile_name_:
                    main(simfile_name_)
                else:
                    sys.exit("Invalid simulation file id - blank id not allowed)...")
    except getopt.GetoptError:
        usage()
# endregion
