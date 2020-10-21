"""This scripts's functions are used to start simulations.

You can start a simulation by executing the following command::

    $ python hive_simulation.py --file=a_simulation_name.json --iters=30

You can also execute all simulation file that exist in
:py:const:`~app.environment_settings.SIMULATION_ROOT` by instead executing::

    $ python hive_simulation.py -d -i 24

If you wish to execute multiple simulations in parallel (to save time) you
can use the -t or --threading flag in either of the previously specified
commands. The threading flag expects an integer that specifies the max
working threads. For example::

    $ python hive_simulation.py -d --iters=1 --threading=12

If you don't have a simulation file yet, run the following instead::

    $ python simfile_generator.py --file=filename.json

Note:
    For the simulation to run without errors you must ensure that:

        1. The specified simulation files exist in \
        :py:const:`~app.environment_settings.SIMULATION_ROOT`.
        2. Any file used by the simulation, e.g., a picture or a .pptx \
        document is accessible in \
        :py:const:`~app.environment_settings.SHARED_ROOT`.
        3. An output file simdirectory exists with default path being: \
        :py:const:`~app.environment_settings.OUTFILE_ROOT`.

"""

import os
import sys
import json
import getopt
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Optional, Dict, Tuple, List

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
    topology = _scenarios[k]["matrices"].pop()
    equilibrium = _scenarios[k]["vectors"].pop()
    return topology, equilibrium
# endregion


# region Module private functions (helpers)
def __makedirs__() -> None:
    """Helper method that reates required simulation working directories if
    they do not exist."""
    os.makedirs(es.SHARED_ROOT, exist_ok=True)
    os.makedirs(es.SIMULATION_ROOT, exist_ok=True)
    os.makedirs(es.OUTFILE_ROOT, exist_ok=True)
    os.makedirs(es.RESOURCES_ROOT, exist_ok=True)


def __can_exec_simfile__(sname: str) -> None:
    """Asserts if simulation can proceed with user specified file.

    Args:
        sname:
            The name of the simulation file, including extension,
            whose existence inside
            :py:const:`~app.environment_settings.SIMULATION_ROOT` will be
            checked.
    """
    spath = os.path.join(es.SIMULATION_ROOT, sname)
    if not os.path.exists(spath):
        sys.exit("Specified simulation file does not exist in SIMULATION_ROOT.")


def __start_simulation__(sname: str, sid: int, epochs: int) -> None:
    """Helper method that orders execution of one simulation instance.

    Args:
        sname:
            The name of the simulation file to be executed.
        sid:
            A sequence number that identifies the simulation execution instance.
        epochs:
            The number of discrete time steps the simulation lasts.
    """
    master_server = class_name_to_obj(
        es.MASTER_SERVERS,
        master_class,
        [sname, sid, epochs, cluster_class, node_class]
    )
    master_server.execute_simulation()


def _parallel_main(threads_count: int,
                   sdir: bool,
                   sname: Optional[str],
                   iters: int,
                   epochs: int) -> None:
    """Helper method that initializes a multi-threaded simulation.

    Args:
        threads_count:
            Number of worker threads that will consume jobs from the Task Pool.
        sdir:
            Indicates whether or not the program will proceed by executing
            all simulations files inside
            :py:const:`~app.environment_settings.SIMULATION_ROOT` folder or
            if will run with the specified file ``sname``.
        sname:
            The name of the simulation file to be executed or ``None`` if
            ``sdir`` is set to ``True``.
        iters:
            How many times each simulation file is executed.
        epochs:
            Number of discrete time steps (epochs) each executed simulation
            lasts.
    """
    with ThreadPoolExecutor(max_workers=threads_count) as executor:
        if sdir:
            snames = list(filter(
                lambda x: "scenarios" not in x, os.listdir(es.SIMULATION_ROOT)))
            for sn in snames:
                for i in range(1, iters + 1):
                    executor.submit(__start_simulation__, sn, i, epochs)
        else:
            __can_exec_simfile__(sname)
            for i in range(1, iters + 1):
                executor.submit(__start_simulation__, sname, i, epochs)


def _single_main(
        sdir: bool, sname: Optional[str], iters: int, epochs: int) -> None:
    """Helper function that initializes a single-threaded simulation.

    Args:
        sdir:
            Indicates whether or not the program will proceed by executing
            all simulations files inside
            :py:const:`~app.environment_settings.SIMULATION_ROOT` folder or
            if will run with the specified file ``sname``.
        sname:
            The name of the simulation file to be executed or ``None`` if
            ``sdir`` is set to ``True``.
        iters:
            How many times each simulation file is executed.
        epochs:
            Number of discrete time steps (epochs) each executed simulation
            lasts.
    """
    if sdir:
        snames = list(filter(
            lambda x: "scenarios" not in x, os.listdir(es.SIMULATION_ROOT)))
        for sn in snames:
            for i in range(1, iters + 1):
                __start_simulation__(sn, i, epochs)
    else:
        __can_exec_simfile__(sname)
        for i in range(1, iters + 1):
            __start_simulation__(sname, i, epochs)
# endregion


def main(threads_count: int,
         sdir: bool,
         sname: Optional[str],
         iters: int,
         epochs: int) -> None:
    """Receives user input and initializes the simulation process.

    Args:
        threads_count:
            Indicates if multiple simulation instances should run in parallel
            (default results in running the simulation in a
            single thread).
        sdir:
            Indicates if the user wishes to execute all simulation files
            that exist in
            :py:const:`~app.environment_settings.SIMULATION_ROOT` or
            if he wishes to run one single simulation file, which must be
            explicitly specified in `sname`.
        sname:
            When `sdir` is set to ``False``, `sname` needs to be specified as a
            non blank string containing the name of the simulation file to
            be executed. The named file must exist in
            :py:const:`~app.environment_settings.SIMULATION_ROOT`.
        iters:
            The number of times the same simulation file should be executed.
        epochs:
            The number of discrete time steps each iteration of each instance
            of a simulation lasts.
    """
    # Creates a MatlabEngineContainer before any thread starts working.
    MatlabEngineContainer.get_instance()

    if threads_count != 0:
        _parallel_main(
            np.abs(threads_count).item(), sdir, sname, iters, epochs)
    else:

        _single_main(sdir, sname, iters, epochs)


if __name__ == "__main__":
    __makedirs__()

    threading = 0
    simdirectory = False
    simfile = None
    iterations = 1
    duration = 480

    master_class = "SGMaster"
    cluster_class = "SGClusterExt"
    node_class = "SGNodeExt"

    short_opts = "df:i:t:e:m:c:n:"
    long_opts = ["directory", "file=", "iters=", "threading=", "epochs=",
                 "master_server=", "cluster_group=", "network_node="]

    try:
        options, args = getopt.getopt(sys.argv[1:], short_opts, long_opts)
        for options, args in options:
            if options in ("-t", "--threading"):
                threading = int(str(args).strip())
            if options in ("-d", "--directory"):
                simdirectory = True
            elif options in ("-f", "--file"):
                simfile = str(args).strip()
                if simfile == "":
                    sys.exit("Simulation file name can not be blank.")
            if options in ("-i", "--iters"):
                iterations = int(str(args).strip())
            if options in ("-e", "--epochs"):
                duration = int(str(args).strip())
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
                 "  --iterations -i (int)\n"
                 "  --epochs -e (int)\n"
                 "  --threading -t (int)\n"
                 "  --directory -d (void)\n"
                 "  --file -f (str)\n"
                 "  --master_server -m (str)\n"
                 "  --cluster_group -c (str)\n"
                 "  --network_node -n (str)\n"
                 "Another cause of error might be a simulation file with "
                 "inconsistent values.")

    if simfile or simdirectory:
        main(threading, simdirectory, simfile, iterations, duration)
    else:
        sys.exit(__err_message__)
