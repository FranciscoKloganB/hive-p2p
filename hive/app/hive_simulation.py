"""This scripts's functions are used to start simulations.

You can start a simulation by executing the following command::

    $ python hive_simulation.py --file=a_simulation_name.json --iterations=30

You can also execute all simulation file that exist in
:py:const:`~app.environment_settings.SIMULATION_ROOT` by instead executing::

    $ python hive_simulation.py -d -i 24

If you wish to execute multiple simulations in parallel (to save time) you
can use the -t or --threading flag in either of the previously specified
commands. The threading flag expects an integer that specifies the max
working threads. For example::

    $ python hive_simulation.py -d --iterations=1 --threading=2

Warning:
    Python's :py:class:`~py:concurrent.futures.ThreadPoolExecutor`
    conceals/supresses any uncaught exceptions, i.e., simulations may fail to
    execute or log items properly and no debug information will be provided

If you don't have a simulation file yet, run the following instead::

    $ python simfile_generator.py --file=filename.json

Note:
    For the simulation to run without errors you must ensure that:

        1. The specified simulation files exist in \
        :py:const:`~app.environment_settings.SIMULATION_ROOT`.
        2. Any file used by the simulation, e.g., a picture or a .pptx \
        document is accessible in \
        :py:const:`~app.environment_settings.SHARED_ROOT`.
        3. An output file directory exists with default path being: \
        :py:const:`~app.environment_settings.OUTFILE_ROOT`.

"""

import os
import sys
import json
import getopt
import traceback
import concurrent.futures

from warnings import warn
from typing import Dict, Tuple, List
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import environment_settings as es

from utils.convertions import class_name_to_obj
from domain.helpers.matlab_utils import MatlabEngineContainer


__err_message__ = ("Invalid arguments. You must specify -f fname or -d, e.g.:\n"
                   "    $ python hive_simulation.py -f simfilename.json\n"
                   "    $ python hive_simulation.py -d")

__log_thread_errors__: bool = True
"""Wether or not the script should crash if a ThreadPoolExecutor 
fails due to an exception and if the exception traceback should be provided."""


# region Sample Scenarios available for debug environments
def __load_scenarios__():
    scenarios = {}
    try:
        scenarios_path = os.path.join(es.RESOURCES_ROOT, "scenarios.json")
        scenarios_file = open(scenarios_path, "r")
        scenarios = json.load(scenarios_file)
        scenarios_file.close()
    except OSError:
        warn(f"Could not load scenarios.json from {es.RESOURCES_ROOT}.\n"
             " > if you need sample scenarios for swarm guidance in your code, "
             "please refer to sample_scenario_generator.py, "
             "otherwise, ignore this warning.")
    return scenarios


_scenarios: Dict[str, Dict[str, List]] = __load_scenarios__()


def get_next_scenario(k: str) -> Tuple[np.ndarray, np.ndarray]:
    """Function used for one-to-one testing of different swarm guidance
    configurations.

    Note:
        This method should only be used when
        :py:const:`app.environment_settings.DEBUG` is set to True.

    Args:
        k:
            A string identifying the pool of matrix, vector pairs to get the
            scenario. Usually, a string representation of an integer which
            corresponds to the network size being tested.

    Returns:
        A topology matrix and a random equilibrium vector that can be used
        to generate Markov chains used for Swarm Guidance.
    """
    if not es.DEBUG:
        warn("get_next_scenario should not be called outside debug envs.")
    topology = np.asarray(_scenarios[k]["matrices"].pop())
    equilibrium = np.asarray(_scenarios[k]["vectors"].pop())
    return topology, equilibrium
# endregion


# region Helpers
def __makedirs__() -> None:
    """Helper method that reates required simulation working directories if
    they do not exist."""
    os.makedirs(es.SHARED_ROOT, exist_ok=True)
    os.makedirs(es.SIMULATION_ROOT, exist_ok=True)
    os.makedirs(es.OUTFILE_ROOT, exist_ok=True)
    os.makedirs(es.RESOURCES_ROOT, exist_ok=True)


def __list_dir__() -> List[str]:
    target_dir = os.listdir(es.SIMULATION_ROOT)
    return list(filter(lambda x: "scenarios" not in x, target_dir))


def _validate_simfile(simfile_name: str) -> None:
    """Asserts if simulation can proceed with user specified file.

    Args:
        simfile_name:
            The name of the simulation file, including extension,
            whose existence inside
            :py:const:`~app.environment_settings.SIMULATION_ROOT` will be
            checked.
    """
    spath = os.path.join(es.SIMULATION_ROOT, simfile_name)
    if not os.path.exists(spath):
        sys.exit(f"The simulation file does not exist in {es.SIMULATION_ROOT}.")


def _simulate(simfile_name: str, sid: int) -> None:
    """Helper method that orders execution of one simulation instance.

    Args:
        simfile_name:
            The name of the simulation file to be executed.
        sid:
            A sequence number that identifies the simulation execution instance.
    """
    master_server = class_name_to_obj(
        es.MASTER_SERVERS, master_class,
        [simfile_name, sid, epochs, cluster_class, node_class]
    )
    master_server.execute_simulation()


def _parallel_main(start: int, stop: int) -> None:
    """Helper method that initializes a multi-threaded simulation.

    Args:
        start:
            A number that marks the first desired identifier for the
            simulations that will execute.
        stop:
            A number that marks the last desired identifier for the
            simulations that will execute. Usually a sum of ``start`` and the
            total number of iterations specified by the user in the scripts'
            arguments.
    """
    with ThreadPoolExecutor(max_workers=threading) as executor:
        futures = []
        if directory:
            for simfile_name in __list_dir__():
                for i in range(start, stop):
                    futures.append(executor.submit(_simulate, simfile_name, i))
        else:
            _validate_simfile(simfile)
            for i in range(start, stop):
                futures.append(executor.submit(_simulate, simfile, i))

        if __log_thread_errors__:
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception:
                    sys.exit(traceback.print_exc())


def _single_main(start: int, stop: int) -> None:
    """Helper function that initializes a single-threaded simulation.

    Args:
        start:
            A number that marks the first desired identifier for the
            simulations that will execute.
        stop:
            A number that marks the last desired identifier for the
            simulations that will execute. Usually a sum of ``start`` and the
            total number of iterations specified by the user in the scripts'
            arguments.
    """
    if directory:
        for simfile_name in __list_dir__():
            for i in range(start, stop):
                _simulate(simfile_name, i)
    else:
        _validate_simfile(simfile)
        for i in range(start, stop):
            _simulate(simfile, i)
# endregion


if __name__ == "__main__":
    __makedirs__()

    directory = False
    simfile = None
    start_iteration = 1
    iterations = 1
    epochs = 480
    threading = 0

    master_class = "SGMaster"
    cluster_class = "SGClusterExt"
    node_class = "SGNodeExt"

    short_opts = "df:i:S:e:t:m:c:n:"
    long_opts = ["directory", "file=",
                 "iterations=", "start_iteration=",
                 "epochs=",
                 "threading=",
                 "master_server=", "cluster_group=", "network_node="]

    try:
        args, values = getopt.getopt(sys.argv[1:], short_opts, long_opts)
        for arg, val in args:
            if arg in ("-d", "--directory"):
                directory = True
            elif arg in ("-f", "--file"):
                simfile = str(val).strip()
            if arg in ("-i", "--iterations"):
                iterations = int(str(val).strip())
            if arg in ("-S", "--start_iteration"):
                start_iteration = int(str(val).strip())
            if arg in ("-e", "--epochs"):
                epochs = int(str(val).strip())
            if arg in ("-t", "--threading"):
                threading = int(str(val).strip())
            if arg in ("-m", "--master_server"):
                master_class = str(val).strip()
            if arg in ("-c", "--cluster_group"):
                cluster_class = str(val).strip()
            if arg in ("-n", "--network_node"):
                node_class = str(val).strip()
    except (getopt.GetoptError, ValueError):
        sys.exit("Execution arguments should have the following data types:\n"
                 "  --directory -d (void)\n"
                 "  --iterations= -i (int)\n"
                 "  --start_iteration= -S (int)\n"
                 "  --epochs= -e (int)\n"
                 "  --threading= -t (int)\n"
                 "  --file= -f (str)\n"
                 "  --master_server= -m (str)\n"
                 "  --cluster_group= -c (str)\n"
                 "  --network_node= -n (str)\n"
                 "Another cause of error might be a simulation file with "
                 "inconsistent values.")

    if simfile is None and not directory:
        sys.exit(__err_message__)
    elif simfile == "" and not directory:
        sys.exit("File name can not be blank. Unless directory option is True.")

    MatlabEngineContainer.get_instance()
    threading = np.ceil(np.abs(threading)).item()

    s = start_iteration
    st = start_iteration + iterations + 1
    _single_main(s, st) if threading in {0, 1} else _parallel_main(s, st)
