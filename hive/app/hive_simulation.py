"""This functionality offered by this module is used to start simulations.

    You can start simulations by executing the following command::

        $ python hive_simulation.py --simfile=filename.json --runs=30

    If you don't have a simulation file yet, run the following instead::

        $ python simulation_file_generator.py --simfile=filename.json

    Notes:
        For the simulation to run without errors you must ensurue that::

            1. You have a specified simulation
"""

import getopt
import os
import sys
from typing import List

import domain.Hivemind as hm
from globals.globals import SIMULATION_ROOT, OUTFILE_ROOT, SHARED_ROOT

err_message = ("Invalid arguments. At least simfile arg must be specified.\n",
               "$ python hive_simulation.py --simfile=simulationfilename.json")


def main(fid, runs):
    if not fid:
        sys.exit("Invalid simulation file name - blank name not allowed)...")

    if not os.path.exists(SHARED_ROOT):
        os.makedirs(SHARED_ROOT)

    if not os.path.exists(SIMULATION_ROOT):
        os.makedirs(SIMULATION_ROOT)

    if not os.path.exists(OUTFILE_ROOT):
        os.makedirs(OUTFILE_ROOT)

    for run in range(runs):
        simulation = hm.Hivemind(simfile_name=fid, sim_id=run)
        simulation.execute_simulation()

    # input_simulation_files: List[str] = os.listdir(SIMULATION_ROOT)
    # for name in input_simulation_files:
    #     for i in range(runs):
    #         simulation = hm.Hivemind(simfile_name=name, sim_id=i+1)
    #         simulation.execute_simulation()


if __name__ == "__main__":
    simfile_name_ = None
    run_ = 30

    try:
        options, args = getopt.getopt(
            sys.argv[1:], "s:r:", ["simfile=", "runs="])

        for options, args in options:
            if options in ("-s", "--simfile"):
                simfile_name_ = str(args).strip()
            if options in ("-r", "--runs"):
                run_ = int(str(args).strip())

        if simfile_name_ and run_:
            main(simfile_name_, run_)
        else:
            print(err_message)

    except getopt.GetoptError:
        print(err_message)
