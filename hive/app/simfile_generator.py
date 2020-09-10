"""This scripts's functions are used to create a simulation file for the user.

You can create a simulation file by following the instructions that
appear in your terminal when running the following command::

    $ python simfile_generator.py --file=filename.json

Note:
    Simulation files are placed inside
    :py:const:`~app.environment_settings.SIMULATION_ROOT` directory. Any file
    used to simulate persistance must be inside
    :py:const:`~app.environment_settings.SHARED_ROOT` directory.

"""
import getopt
import itertools
import json
import os
import string
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy

from environment_settings import SHARED_ROOT, SIMULATION_ROOT
from scripts.python import normal_distribution_generator as ng


# region Input Consumption and Verification
from utils.convertions import truncate_float_value


def _input_character_option(message: str, white_list: List[str]) -> str:
    """Obtains a user inputed character within a predefined set.

    Args:
        message:
            The message to be printed to the user upon first input request.
        white_list:
            A list of valid option characters.

    Returns:
        The character that represents the initial distribution of files in a
        :py:mod:`domain.cluster_groups`'s class instance desired by the user.
    """
    character = input(message)
    while True:
        if character in white_list:
            return character
        character = input(f"Choose an option among {white_list}. Try again: ")


def _input_bounded_integer(
        message: str, lower_bound: int = 2, upper_bound: int = 10000000) -> int:
    """Obtains a user inputed integer within the specified closed interval.

    Args:
        message:
            The message to be printed to the user upon first input request.
        lower_bound:
             Any input equal or smaller than `lower_bound` is
            rejected.
        upper_bound:
             Any input equal or bigger than `upper_bound` is rejected.

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


def _input_bounded_float(
        message: str, lower_bound: float = 0.0, upper_bound: float = 100.0
) -> float:
    """Obtains a user inputed integer within the specified closed interval.

    Args:
        message:
            The message to be printed to the user upon first input request.
        lower_bound:
             Any input smaller than`lower_bound` is rejected.
        upper_bound:
             Any input bigger than `upper_bound` is rejected.

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


def _input_filename(message: str) -> str:
    """Asks the user to input the name of a file in the command line terminal.

    A warning message is displayed if the specified file does not exist inside
    :py:const:`~app.environment_settings.SHARED_ROOT`

    Note:
        Defaults to ``"FBZ_0134.NEF"`` when input is blank. This file should
        be present inside :py:const:`~app.environment_settings.SHARED_ROOT`
        unless it was previously deleted by the user.

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
            print(f"{file_name} is not inside ~/cluster/app/static/shared folder.")
        return file_name


def _in_yes_no(message: str) -> bool:
    """Asks the user to reply with yes or no to a message.

    Args:
        message:
            The message to be printed to the user upon first input request.

    Returns:
        ``True`` if user presses yes, otherwise ``False``.
    """
    char = input(f"{message} [y/n]: ").lower()
    while True:
        if char == 'y':
            return True
        elif char == 'n':
            return False
        else:
            char = input("Press 'y' for yes or 'n' for no. Try again: ")
# endregion


# region Helpers
def yield_label() -> str:
    """Used to generate an arbrirary numbers of unique labels.

        Examples:
            The following code snippets illustrate the result of calling this
            method ``n`` times. ::

                >>> n = 4
                >>> for s in itertools.islice(yield_label(), n):
                ...     return s
                [a, b, c, d]

               >>> n = 4 + 26
                >>> for s in itertools.islice(yield_label(), n):
                ...     return s
                [a, b, c, d, ..., aa, ab, ac, ad]

    Yields:
        The next string label in the sequence.
    """
    for size in itertools.count(1):
        for s in itertools.product(string.ascii_lowercase, repeat=size):
            yield "".join(s)


def _init_nodes_uptime() -> Dict[str, float]:
    """Creates a record containing network nodes' uptime.

    Returns:
        A dictionary where keys are
        :py:attr:`network node identifiers <app.domain.network_nodes.Node.id>`
        and values are their respective uptimes
        :py:attr:`uptime <app.domain.network_nodes.Node.uptime>` values.
    """
    number_of_nodes = _input_bounded_integer("Network Size [2, 10000000]: ")

    min_uptime = _input_bounded_float("Min node uptime [0.0, 100.0]: ") / 100
    min_uptime = truncate_float_value(min_uptime, 6)

    max_uptime = _input_bounded_float("Max node uptime [0.0, 100.0]: ") / 100
    max_uptime = truncate_float_value(max_uptime, 6)

    mean = _input_bounded_float("Distribution mean [0.0, 100.0]: ")
    std = _input_bounded_float("Standard deviation [0.0, 100.0]: ")

    samples = ng.generate_samples(
        surveys=1, sample_count=number_of_nodes, mean=mean, std=std).tolist()

    nodes_uptime = {}
    for label in itertools.islice(yield_label(), number_of_nodes):
        uptime = numpy.abs(samples.pop()[0]) / 100.0
        uptime = numpy.clip(uptime, min_uptime, max_uptime)
        nodes_uptime[label] = truncate_float_value(uptime.item(), 6)
    samples.clear()

    return nodes_uptime


def _init_persisting_dict() -> Dict[str, Any]:
    """Creates the "persisting" key of simulation file.

    Returns:
        A dictionary containing data respecting files to be shared in the system
    """
    persisting: Dict[str, Any] = {}

    print(
        "\nAny file you want to simulate persistance of must be inside the "
        "following folder: ~/cluster/app/static/shared\n"
        "You may also want to keep a backup of such file in:  "
        "~/cluster/app/static/shared/shared_backups"
    )

    add_file: bool = True
    while add_file:
        file_name = _input_filename(
            "Name the file (with extension) you wish to simulate persistence of: ")

        options_message = ("\nSelect how files blocks are spread across "
                           "clusters at the start of the simulation: {\n"
                           "   u: uniform distribution among network nodes,\n"
                           "   i: ideal distribution, e.g., near a steady-state vector, \n"
                           "   a: all replicas given to N different nodes,\n"
                           "   o: each network node receives one random replica\n"
                           "}: ")
        options_list = ["u", "U", "i", "I", "a", "A", "o", "O"]
        option_choice = _input_character_option(options_message, options_list)

        persisting[file_name] = {}
        persisting[file_name]["spread"] = option_choice.lower()
        persisting[file_name]["cluster_size"] = _input_bounded_integer(
            "\nNumber of nodes that should be sharing the next file: ")

        add_file = _in_yes_no(
            "\nSimulate persistence of another file in simulation?")

    return persisting
# endregion


def main(simfile_name: str) -> None:
    """Creates a JSON file within the user's file system that is used by
    :py:mod:`hive_simulation`.

    Note:
        The name of the created file concerns the name of the simulation file.
        It does not concern the name or names of the files whose persistence
        is being simulated.

    Args:
        simfile_name:
            Name to be assigned to JSON file in the user's file system.
    """
    simfile_json: Dict[str, Any] = {
        "nodes_uptime": _init_nodes_uptime(),
        "persisting": _init_persisting_dict()
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
