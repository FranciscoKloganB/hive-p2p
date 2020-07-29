import getopt
import os
import sys
from typing import List

import domain.Hivemind as hm
# region Usage, Help and Main
from globals.globals import SIMULATION_ROOT, OUTFILE_ROOT


def usage():
    print("---------------------------------------------------------------\n",
          "Run a simulation for Markov Chain Based Swarm Guidance algorithm ",
          "on a P2P Network that persists files... Example usage:\n",
          "python hive_simulation.py --simfile=simulationfilename.json\n"
          "---------------------------------------------------------------\n")
    sys.exit(0)


def myhelp():
    print("---------------------------------------------------------------\n",
          "To create a simulation file automatically use the script: \n",
          "simulation_file_generator.py script in ~/scripts/python folder.\n",
          "---------------------------------------------------------------\n")
    sys.exit(0)


def main(fname, runs_per_input_file):
    if not fname:
        sys.exit("Invalid simulation file name - blank name not allowed)...")

    if not os.path.exists(OUTFILE_ROOT):
        os.makedirs(OUTFILE_ROOT)

    input_simulation_files: List[str] = os.listdir(SIMULATION_ROOT)
    for name in input_simulation_files:
        for i in range(runs_per_input_file):
            simulation = hm.Hivemind(simfile_name=name, sim_number=i+1)
            simulation.execute_simulation()
    # simulation = hm.Hivemind(
    #     simfile_name=input_simulation_files[0], sim_number=1337)
    # simulation.execute_simulation()
# endregion


if __name__ == "__main__":
    simfile_name_ = None
    run_ = 30
    try:
        options, args = getopt.getopt(
            sys.argv[1:], "uhs:r:", ["usage", "help", "simfile=", "runs="])

        for options, args in options:
            if options in ("-u", "--usage"):
                usage()
            if options in ("-h", "--help"):
                myhelp()
            if options in ("-s", "--simfile"):
                simfile_name_ = str(args).strip()
                print(simfile_name_)
            if options in ("-r", "--runs"):
                run_ = int(str(args).strip())
                print(run_)
        if simfile_name_ and run_:
            main(simfile_name_, run_)
    except getopt.GetoptError:
        usage()
