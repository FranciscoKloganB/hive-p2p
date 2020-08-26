"""This scripts's functions are used to start simulations.

You can start a simulation by executing the following command::

    $ python hive_simulation.py --file=a_simulation_name.json --iters=30

You can also execute all simulation file that exist in
:py:const:`environment_settings.SIMULATION_ROOT` by instead executing::

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
        :py:const:`app.environment_settings.SIMULATION_ROOT`.
        2. Any file used by the simulation, e.g., a picture or a .pptx \
        document is accessible in \
        :py:const:`app.environment_settings.SHARED_ROOT`.
        3. An output file simdirectory exists with default path being: \
        :py:const:`app.environment_settings.OUTFILE_ROOT`.

"""

import getopt
import os
import sys
from concurrent.futures.thread import ThreadPoolExecutor

import numpy

from domain.helpers.matlab_utils import MatlabEngineContainer
from environment_settings import SIMULATION_ROOT, OUTFILE_ROOT, SHARED_ROOT, \
    MASTER_SERVERS

__err_message__ = ("Invalid arguments. You must specify -f fname or -d, e.g.:\n"
                   "    $ python hive_simulation.py -f simfilename.json\n"
                   "    $ python hive_simulation.py -d")


# region Module private functions (helpers)
from utils.convertions import class_name_to_obj


def __makedirs() -> None:
    """Creates required simulation working directories if they do not exist."""
    if not os.path.exists(SHARED_ROOT):
        os.makedirs(SHARED_ROOT)

    if not os.path.exists(SIMULATION_ROOT):
        os.makedirs(SIMULATION_ROOT)

    if not os.path.exists(OUTFILE_ROOT):
        os.makedirs(OUTFILE_ROOT)


def __can_exec_simfile(sname: str) -> None:
    """Verifies if input simulation file name exists in
    :py:const:`app.environment_settings.SIMULATION_ROOT`."""
    spath = os.path.join(SIMULATION_ROOT, sname)
    if not os.path.exists(spath):
        sys.exit("Specified simulation file does not exist in SIMULATION_ROOT.")


def __start_simulation(sname: str, sid: int, epochs: int) -> None:
    """Executes one instance of the simulation

    Args:
        sname:
            The name of the simulation file to be executed.
        sid:
            A sequence number that identifies the simulation execution instance.
        epochs:
            The number of discrete time steps the simulation lasts.
    """
    master_server = class_name_to_obj(
        MASTER_SERVERS,
        master_class,
        [sname, sid, epochs, cluster_class, node_class]
    )
    master_server.execute_simulation()


def __parallel_main(
        threads_count: int, sdir: bool, sname: str, iters: int, epochs: int
) -> None:
    """Helper method that initializes a multi-threaded simulation."""
    with ThreadPoolExecutor(max_workers=threads_count) as executor:
        if sdir:
            snames = os.listdir(SIMULATION_ROOT)
            for sn in snames:
                for i in range(iters):
                    executor.submit(__start_simulation, sn, i, epochs)
        else:
            __can_exec_simfile(sname)
            for i in range(iters):
                executor.submit(__start_simulation, sname, i, epochs)


def __single_main(sdir: bool, sname: str, iters: int, epochs: int) -> None:
    """Helper function that initializes a single-threaded simulation."""
    if sdir:
        snames = os.listdir(SIMULATION_ROOT)
        for sn in snames:
            for i in range(iters):
                __start_simulation(sn, i, epochs)
    else:
        __can_exec_simfile(sname)
        for i in range(iters):
            __start_simulation(sname, i, epochs)
# endregion


def main(
        threads_count: int, sdir: bool, sname: str, iters: int, epochs: int
) -> None:
    """Receives user input and initializes the simulation process.

    Args:
        threads_count:
            Indicates if multiple simulation instances should run in parallel
            (default results in running the simulation in a
            single thread).
        sdir:
            Indicates if the user wishes to execute all simulation files
            that exist in :py:const:`environment_settings.SIMULATION_ROOT` or
            if he wishes to run one single simulation file, which must be
            explicitly specified in `sname`.
        sname:
            When `sdir` is set to False, `sname` needs to be specified as a
            non blank string containing the name of the simulation file to
            be executed. The named file must exist in
            :py:const:`environment_settings.SIMULATION_ROOT`.
        iters:
            The number of times the same simulation file should be executed.
        epochs:
            The number of discrete time steps each iteration of each instance
            of a simulation lasts.
    """
    MatlabEngineContainer.get_instance()

    if threads_count != 0:
        __parallel_main(
            numpy.abs(threads_count).item(), sdir, sname, iters, epochs)
    else:
        __single_main(sdir, sname, iters, epochs)


if __name__ == "__main__":
    __makedirs()

    threading = 0
    simdirectory = False
    simfile = None
    iterations = 30
    duration = 720

    master_class = "Master"
    cluster_class = "HiveCluster"
    node_class = "HiveNode"

    try:
        short_opts = "df:i:t:e:m:c:n:"
        long_opts = [
            "directory", "file=", "iters=", "threading=", "epochs=",
            "master_server=", "cluster_group=", "network_node="
        ]
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

        if simfile or simdirectory:
            main(threading, simdirectory, simfile, iterations, duration)
        else:
            sys.exit(__err_message__)

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
                 "  --network_node -n (str)\n")
