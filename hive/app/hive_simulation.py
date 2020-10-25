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

from warnings import warn
from typing import Optional, Dict, Tuple, List
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import environment_settings as es

from utils.convertions import class_name_to_obj
from domain.helpers.matlab_utils import MatlabEngineContainer


__err_message__ = ("Invalid arguments. You must specify -f fname or -d, e.g.:\n"
                   "    $ python hive_simulation.py -f simfilename.json\n"
                   "    $ python hive_simulation.py -d")


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


def _start_simulation(simfile_name: str, sid: int) -> None:
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
    """Helper method that initializes a multi-threaded simulation."""
    with ThreadPoolExecutor(max_workers=threading) as executor:
        if directory:
            for simfile_name in __list_dir__():
                for i in range(start, stop):
                    executor.submit(_start_simulation, simfile_name, i, epochs)
        else:
            _validate_simfile(simfile)
            for i in range(start, stop):
                executor.submit(_start_simulation, simfile, i, epochs)


def _single_main(start: int, stop: int) -> None:
    """Helper function that initializes a single-threaded simulation."""
    if directory:
        for simfile_name in __list_dir__():
            for i in range(start, stop):
                _start_simulation(simfile_name, i)
    else:
        _validate_simfile(simfile)
        for i in range(start, stop):
            _start_simulation(simfile, i)
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
        options, args = getopt.getopt(sys.argv[1:], short_opts, long_opts)
        for options, args in options:
            if options in ("-d", "--directory"):
                directory = True
            elif options in ("-f", "--file"):
                simfile = str(args).strip()

            if options in ("-i", "--iterations"):
                iterations = int(str(args).strip())
            if options in ("-S", "--start_iteration"):
                start_iteration = int(str(args).strip())
            if options in ("-e", "--epochs"):
                epochs = int(str(args).strip())
            if options in ("-t", "--threading"):
                threading = int(str(args).strip())
            if options in ("-m", "--master_server"):
                master_class = str(args).strip()
            if options in ("-c", "--cluster_group"):
                cluster_class = str(args).strip()
            if options in ("-n", "--network_node"):
                node_class = str(args).strip()
    except getopt.GetoptError:
        sys.exit(__err_message__)
    except ValueError:
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

    if not simfile or not directory:
        sys.exit(__err_message__)

    if not directory and simfile == "":
        sys.exit("File name can not be blank. Unless directory option is True.")

    MatlabEngineContainer.get_instance()
    threading = np.ceil(np.abs(threading)).item()

    s = start_iteration
    st = start_iteration + iterations + 1
    _single_main(s, st) if threading in (0, 1) else _parallel_main(s, st)
