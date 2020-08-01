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


def multithreaded_main(simfile: str, iters: int) -> None:
    for run in range(iters):
        hm.Hivemind(simfile_name=simfile, sid=run).execute_simulation()


def execute_simulation(sname: str, sid: int) -> None:
    """Executes one instance of the simulation

    Args:
        sname:
            The name of the simulation file to be executed.
        sid:
            An integer that uniquely identifies the simulation execution.
    """
    hm.Hivemind(simfile_name=sname, sid=sid).execute_simulation()


def __makedirs__() -> None:
    """Creates required simulation working directories"""
    if not os.path.exists(SHARED_ROOT):
        os.makedirs(SHARED_ROOT)

    if not os.path.exists(SIMULATION_ROOT):
        os.makedirs(SIMULATION_ROOT)

    if not os.path.exists(OUTFILE_ROOT):
        os.makedirs(OUTFILE_ROOT)


def __can_exec_simfile__(sname: str) -> None:
    """Verifies if input simulation file name exists in working simdirectory."""
    if sname == "":
        sys.exit("Simulation file name can not be blank.")

    spath = os.path.join(SIMULATION_ROOT, sname)
    if not os.path.exists(spath):
        sys.exit("Specified simulation file does not exist in SIMULATION_ROOT.")


def main(sdir: bool, multithread: bool, sname: str, iters: int) -> None:
    __makedirs__()
    __can_exec_simfile__(sname)

    if not multithread:
        for i in range(iters):
            execute_simulation(sname, i)
        # input_simulation_files: List[str] = os.listdir(SIMULATION_ROOT)
        # for name in input_simulation_files:
        #     for i in range(iters):
        #         execute_simulation(name, i)
        return

    multithreaded_main(sname, iters)
    return


if __name__ == "__main__":
    threading = False
    simdirectory = False
    simfile = None
    iterations = 30

    try:
        options, args = getopt.getopt(sys.argv[1:], "tdf:r:", ["threading",
                                                               "directory",
                                                               "file=",
                                                               "iters="])

        for options, args in options:
            if options in ("-t", "--threading"):
                threading = True

            if options in ("-d", "--directory"):
                simdirectory = True
            elif options in ("-f", "--file"):
                simfile = str(args).strip()

            if options in ("-i", "--iters"):
                iterations = int(str(args).strip())

        if simfile or simdirectory:
            main(simdirectory, threading, simfile, iterations)
        else:
            print(err_message)

    except getopt.GetoptError:
        print(err_message)
