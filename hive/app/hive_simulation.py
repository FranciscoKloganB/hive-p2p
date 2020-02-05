import getopt
import json
import os
import sys


# region Usage, Help and Main
def usage():
    print(" -------------------------------------------------------------------------")
    print(" Francisco Barros (francisco.teixeira.de.barros@tecnico.ulisboa.pt\n")
    print(" Run a simulation for Markov Chain Based Swarm Guidance algorithm on a P2P Network that persists files\n")
    print(" Typical usage: python hive_simulation.py --simfile=simulationfilename.json\n")
    print(" Display all optional flags and other important notices: hive_simulation.py --help\n")
    print(" -------------------------------------------------------------------------\n")
    sys.exit(" ")


def myhelp():
    with open("{}/static/simfiles/simfile_example.json".format(os.getcwd())) as json_file:
        print("-------------------------------------------------------------------------\n")
        print("To create a simulation file automatically use simulation_file_generator.py script in ~/scripts/python folder.\n")
        print("-------------------------------------------------------------------------\n")
        print("If you wish to manually create a simulation file here is an example of its structure:\n")
        print("-------------------------------------------------------------------------\n")
        print(json.dumps(json.load(json_file), indent=4, sort_keys=True))
        print(" -------------------------------------------------------------------------\n")
    sys.exit(" ")


def main(simfile_name):
    if not simfile_name:
        sys.exit("Invalid simulation file name - blank name not allowed)...")

    from domain.Hivemind import Hivemind

    simulation = Hivemind(simfile_name)
    simulation.execute_simulation()
# endregion


if __name__ == "__main__":
    simfile_name_ = None
    try:
        options, args = getopt.getopt(sys.argv[1:], "uhs:", ["usage", "help", "simfile="])
        for options, args in options:
            if options in ("-u", "--usage"):
                usage()
            if options in ("-h", "--help"):
                myhelp()
            if options in ("-s", "--simfile"):
                simfile_name_ = str(args).strip()
                main(simfile_name_)
    except getopt.GetoptError:
        usage()
