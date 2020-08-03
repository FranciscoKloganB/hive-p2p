"""This scripts's functions are used to create a simulation file for the user.

    You can create a simulation file by following the instructions that
    appear in your terminal when running the following command::

        $ python simfile_generator.py --file=filename.json

    Notes:
        Simulation files are placed in:
        :py:const:`~environment_settings.SIMULATION_ROOT`.

        Any file used to simulate persistance must be in:
        :py:const:`~environment_settings.SHARED_ROOT`.
"""
import getopt
import itertools
import json
import logging
import math
import os
import string
import sys
import numpy

from pathlib import Path
from typing import List, Dict, Any

from environment_settings import SHARED_ROOT, SIMULATION_ROOT
from scripts.python import normal_distribution_generator as ng


# region Input Consumption and Verification
def __input_character_option(message: str, white_list: List[str]) -> str:
    """Obtains a user inputed character within a predefined set.

    Args:
        message:
            The message to be printed to the user upon first input request.
        white_list:
            A list of valid option characters.

    Returns:
        The character that represents the initial distribution of files in a
        :py:mod:`~domain.cluster_groups`'s class instance desired by the user.
    """
    character = input(message)
    while True:
        if character in white_list:
            return character
        character = input(f"Choose an option among {white_list}. Try again: ")


def __input_bounded_integer(
        message: str, lower_bound: int = 2, upper_bound: int = 16384) -> int:
    """Obtains a user inputed integer within the specified closed interval.

    Args:
        message:
            The message to be printed to the user upon first input request.
        lower_bound:
            Any input equal or smaller than`lower_bound` is rejected (
            default is 2).
        upper_bound:
            Any input equal or bigger than `upper_bound` is rejected (
            default is 16384).

    Returns:
        An integer inputed by the user.
    """
    integer = input(message)
    while True:
        try:
            integer = int(integer)
            if lower_bound <= integer <= upper_bound:
                return integer
            integer = input(f"Input should be in [{lower_bound}, "
                            f"{upper_bound}]. Try again: ")
        except ValueError:
            integer = input("Input should be a integer. Try again: ")
            continue


def __input_bounded_float(
        message: str, lower_bound: float = 0.0, upper_bound: float = 100.0
) -> float:
    """Obtains a user inputed integer within the specified closed interval.

    Args:
        message:
            The message to be printed to the user upon first input request.
        lower_bound:
            optional; Any input smaller than`lower_bound` is rejected (
            default is 0.0).
        upper_bound:
            optional; Any input bigger than `upper_bound` is rejected (
            default is 100.0).

    Returns:
        An float inputed by the user.
    """
    double = input(message)
    while True:
        try:
            double = float(double)
            if lower_bound <= double <= upper_bound:
                return double
            double = input(f"Input should be in [{lower_bound}, "
                           f"{upper_bound}]. Try again: ")
        except ValueError:
            double = input("Input should be a float. Try again: ")
            continue


def __input_filename(message: str) -> str:
    """Verifies if inputed file name exists in
    :py:const:`~environment_settings.SHARED_ROOT` directory.

    Note:
        If a blank file name is given, default value of FBZ_0134.NEF is
        selected.

    Args:
        message:
            The message to be printed to the user upon first input request.

    Returns:
        A file name with extension.
    """
    file_name = input(message).strip()
    while True:
        if file_name == "":
            print("Invalid name, falling back to default 'FBZ_0134.NEF'.")
            file_name = "FBZ_0134.NEF"
        if not Path(os.path.join(SHARED_ROOT, file_name)).is_file():
            print(f"{file_name} is not inside ~/hive/app/static/shared folder.")
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
# endregion


# region Generation Functions
def __truncate_float_value(f: float, d: int) -> float:
    """Truncates a float value without rounding.

    Args:
        f:
            The float value to truncate.
        d:
            The number of decimal places the float can have.

    Returns:
        The truncated float.
    """
    return math.floor(f * 10 ** d) / 10 ** d


def __yield_label() -> str:
    """Used to generate an arbrirary numbers of unique labels.

    Yields:
        The next string label in the sequence.

    Examples:
        >>> n = 4
        >>> for s in itertools.islice(__yield_label(), n):
        ...     return s
        [a, b, c, d]

       >>> n = 4 + 26
        >>> for s in itertools.islice(__yield_label(), n):
        ...     return s
        [a, b, c, d, ..., aa, ab, ac, ad]
    """
    for size in itertools.count(1):
        for s in itertools.product(string.ascii_lowercase, repeat=size):
            yield "".join(s)


def __init_nodes_uptime() -> Dict[str, float]:
    """Creates a record containing network nodes' uptime.

    Returns:
        A collection mapping :py:mod:`~domain.network_nodes`'s class
        instance labels to their respective uptime values.
    """
    number_of_nodes = __input_bounded_integer("Network Size [2, 16384]: ")

    min_uptime = __input_bounded_float("Min node uptime [0.0, 100.0]: ") / 100
    min_uptime = __truncate_float_value(min_uptime, 6)

    max_uptime = __input_bounded_float("Max node uptime [0.0, 100.0]: ") / 100
    max_uptime = __truncate_float_value(max_uptime, 6)

    mean = __input_bounded_float("Distribution mean [0.0, 100.0]: ")
    std = __input_bounded_float("Standard deviation [0.0, 100.0]: ")

    samples = ng.generate_samples(surveys=1, mean=mean, std=std).tolist()

    nodes_uptime = {}
    for label in itertools.islice(__yield_label(), number_of_nodes):
        # gets and removes last element in samples to assign it to label
        uptime = numpy.abs(samples.pop()[0]) / 100.0
        uptime = numpy.clip(uptime, min_uptime, max_uptime)
        nodes_uptime[label] = __truncate_float_value(uptime.item(), 6)
    samples.clear()

    return nodes_uptime


def __init_persisting_dict() -> Dict[str, Any]:
    """Creates the "persisting" key of simulation file.

    Returns:
        A dictionary containing data respecting files to be shared in the system
    """
    persisting: Dict[str, Any] = {}

    print(
        "\nAny file you want to simulate persistance of must be inside the "
        "following folder: ~/hive/app/static/shared\n"
        "You may also want to keep a backup of such file in:  "
        "~/hive/app/static/shared/shared_backups"
    )

    add_file: bool = True
    while add_file:
        file_name = __input_filename(
            "Name the file (with extension) you wish to simulate persistence of: ")

        options_message = ("\nSelect how files blocks are spread across "
                           "clusters at the start of the simulation: {\n"
                           "   u: uniform distribution among network nodes,\n"
                           "   i: near steady-state distribution,\n"
                           "   a: all files concentrated on N replicas\n}\n")
        options_list = ["u", "U", "i", "I", "a", "A"]
        option_choice = __input_character_option(options_message, options_list)

        persisting[file_name] = {}
        persisting[file_name]["spread"] = option_choice.lower()
        persisting[file_name]["cluster_size"] = __input_bounded_integer(
            "Number of nodes that should be sharing the next file: \n")

        add_file = __in_yes_no(
            "\nSimulate persistence of another file in simulation?")

    return persisting
# endregion


def main(simfile_name: str) -> None:
    """Creates a JSON file within the user's file system that is used by
    :py:mod:`hive_simulation`.

    Note:
        The name of the created file concerns the name of the simulation file.
        It does not concern the name or names of the files whose persistence
        is being simulalted.

    Args:
        simfile_name:
            Name to be assigned to JSON file in the user's file system.
    """
    simfile_json: Dict[str, Any] = {
        "nodes_uptime": __init_nodes_uptime(),
        "persisting": __init_persisting_dict()
    }

    with open(os.path.join(SIMULATION_ROOT, simfile_name), 'w') as outfile:
        json.dump(simfile_json, outfile, indent=4)


if __name__ == "__main__":
    simfile_name_: str = ""

    try:
        short_opts = "f:"
        long_opts = ["file="]
        options, args = getopt.getopt(sys.argv[1:], short_opts, long_opts)
        for options, args in options:
            if options in ("-f", "--file"):
                simfile_name_ = str(args).strip()
                if simfile_name_:
                    main(simfile_name_)
                else:
                    sys.exit("Invalid simulation file - blank id not allowed")

    except getopt.GetoptError:
        print("Usage: python simfile_generator.py --file=filename.json")
