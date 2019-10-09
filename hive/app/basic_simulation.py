import getopt
import json
import os
import sys


def usage():
    print(" -------------------------------------------------------------------------")
    print(" Francisco Barros (francisco.teixeira.de.barros@tecnico.ulisboa.pt\n")
    print(" Run a simulation for Markov Chain Based Swarm Guidance algorithm on a P2P Network that persists files\n")
    print(" Typical usage: basic_simulation.py --simfile=sim01.json\n")
    print(" Display all optional flags and other important notices: basic_simulation.py --help\n")
    print(" -------------------------------------------------------------------------\n")
    sys.exit(" ")


def help():
    with open("{}/static/simfiles/simfile_example.json".format(os.getcwd())) as json_file:
        print("-------------------------------------------------------------------------\nSimulation file example:\n")
        print(json.dumps(json.load(json_file), indent=4, sort_keys=True))
        print(" -------------------------------------------------------------------------\n")
    sys.exit(" ")


def main():

    try:
        options, args = getopt.getopt(sys.argv[1:], "uhs:", ["usage", "help", "simfile="])
        for options, args in options:
            if options in ("-u", "--usage"):
                usage()
            if options in ("-h", "--help"):
                help()
            if options in ("-s", "--simfile"):
                sim_file_path = str(args).strip()
                if not sim_file_path:
                    sys.exit("Invalid simulation filepath. A simulation file is required for execution rules.")

    except getopt.GetoptError:
        usage()


if __name__ == "__main__":
    main()
