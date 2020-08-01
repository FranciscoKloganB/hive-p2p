"""This functionality offered by this module is used to start simulations.

    You can start simulations by executing the following command::

        $ python hive_simulation.py --file=filename.json --iters=30

    If you don't have a simulation file yet, run the following instead::

        $ python simulation_file_generator.py --file=filename.json

    Notes:
        For the simulation to run without errors you must ensurue that::
            1. The specified simulation file exists in
            :py:const:`~globals.globals.SIMULATION_ROOT`.

            2. Any file used by the simulation, e.g., a picture or a .pptx
            document is accessible in :py:const:`~globals.globals.SHARED_ROOT`.

            3. An output file simdirectory exists with default path being:
            :py:const:`~globals.globals.OUTFILE_ROOT`.
"""

import getopt
import os
import sys

import domain.Hivemind as hm
from globals.globals import SIMULATION_ROOT, OUTFILE_ROOT, SHARED_ROOT

err_message = ("Invalid arguments. You must specify -f or -d running options\n",
               "$ python hive_simulation.py -f simfilename.json\nor\n",
               "$ python hive_simulation.py -d")


# region Module Private Functions
def __makedirs() -> None:
    """Creates required simulation working directories if they do not exist."""
    if not os.path.exists(SHARED_ROOT):
        os.makedirs(SHARED_ROOT)

    if not os.path.exists(SIMULATION_ROOT):
        os.makedirs(SIMULATION_ROOT)

    if not os.path.exists(OUTFILE_ROOT):
        os.makedirs(OUTFILE_ROOT)


def __can_exec_simfile(sname: str) -> None:
    """Verifies if input simulation file name exists in ~/*/SIMULATION_ROOT"""
    spath = os.path.join(SIMULATION_ROOT, sname)
    if not os.path.exists(spath):
        sys.exit("Specified simulation file does not exist in SIMULATION_ROOT.")


def __execute_simulation(sname: str, sid: int) -> None:
    """Executes one instance of the simulation

    Args:
        sname:
            The name of the simulation file to be executed.
        sid:
            A sequence number that identifies the simulation execution instance.
    """
    hm.Hivemind(sname, sid).execute_simulation()
# endregion


def multi_threaded_main(sdir: bool, sname: str, iters: int) -> None:
    if sdir:
        snames = os.listdir(SIMULATION_ROOT)
        for sn in snames:
            for i in range(iters):
                __execute_simulation(sn, i)
    else:
        __can_exec_simfile(sname)
        for i in range(iters):
            __execute_simulation(sname, i)


def single_threaded_main(sdir, sname, iters):
    if sdir:
        snames = os.listdir(SIMULATION_ROOT)
        for sn in snames:
            for i in range(iters):
                __execute_simulation(sn, i)
    else:
        __can_exec_simfile(sname)
        for i in range(iters):
            __execute_simulation(sname, i)


def main(multithread: bool, sdir: bool, sname: str, iters: int) -> None:
    """Receives user input and initializes the simulation process.

    Args:
        multithread:
            Indicates if multiple simulation instances should run in parallel.
        sdir:
            Indicates if the user wishes to execute all simulation files
            that exist in :py:const:`~globals.globals.SIMULATION_ROOT` or
            if he wishes to run one single simulation file, which must be
            explicitly specified in `sname`.
        sname:
            When `sdir` is set to False, `sname` needs to be specified as a
            non blank string containing the name of the simulation file to
            be executed. The named file must exist in
            :py:const:`~globals.globals.SIMULATION_ROOT`.
        iters:
            The number of times the same simulation file should be executed.
    """
    if multithread:
        multi_threaded_main(sdir, sname, iters)
    else:
        single_threaded_main(sdir, sname, iters)


if __name__ == "__main__":
    __makedirs()

    threading = False
    simdirectory = False
    simfile = None
    iterations = 30

    try:
        short_opts = "tdf:i:"
        long_opts = ["threading", "directory", "file=", "iters="]
        options, args = getopt.getopt(sys.argv[1:], short_opts, long_opts)

        for options, args in options:
            if options in ("-t", "--threading"):
                threading = True
            if options in ("-d", "--directory"):
                simdirectory = True
            elif options in ("-f", "--file"):
                simfile = str(args).strip()
                if simfile == "":
                    sys.exit("Simulation file name can not be blank.")
            if options in ("-i", "--iters"):
                iterations = int(str(args).strip())

        if simfile or simdirectory:
            main(threading, simdirectory, simfile, iterations)
        else:
            sys.exit(err_message)

    except getopt.GetoptError:
        sys.exit(err_message)
