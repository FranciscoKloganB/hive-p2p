"""This is a non-essential module used for convex optimization prototyping.

This functionality tests and compares the mixing rate of various
markov matrices.

    You can start a test by executing the following command::

        $ python mixing_rate_sampler.py --samples=1000

    You can also specify the names of the functions used to generate markov
    matrices like so::

        $ python mixing_rate_sampler.py -s 10 -f afunc,anotherfunc,yetanotherfunc

        Default functions set {
            "new_mh_transition_matrix",
            "new_sdp_mh_transition_matrix",
            "new_go_transition_matrix",
            "new_mgo_transition_matrix"
        }


"""

from __future__ import annotations

import collections
import getopt
import importlib
import json
import os
import sys
from typing import List, Any, OrderedDict

import numpy as np
from cvxpy.error import SolverError
from matlab.engine import EngineError

import domain.helpers.matrices as mm
from domain.helpers.matlab_utils import MatlabEngineContainer
from environment_settings import MIXING_RATE_SAMPLE_ROOT

_SizeResultsDict: OrderedDict[str, List[float]]
_ResultsDict: OrderedDict[str, _SizeResultsDict]


def main():
    """Compares the mixing rate of the markov matrices generated by all
    specified `functions`, `samples` times.

    The execution of the main method results in a JSON file outputed to
    :py:const:`environment_settings.MIXING_RATE_SAMPLE_ROOT` folder.
    """
    if not os.path.exists(MIXING_RATE_SAMPLE_ROOT):
        os.makedirs(MIXING_RATE_SAMPLE_ROOT)

    MatlabEngineContainer.get_instance()

    results: _ResultsDict = collections.OrderedDict()

    size = 8
    while size <= max_adj_size:
        print(f"\nTesting matrices of size: {size}.")

        size_results: _SizeResultsDict = collections.OrderedDict()
        for name in functions:
            size_results[name] = []

        for i in range(1, samples + 1):
            print(f"    Sample {i}.")
            a = np.asarray(mm.new_symmetric_matrix(size))
            a = mm.make_connected(a)
            v_ = np.abs(np.random.randn(size))
            v_ /= v_.sum()

            for name in functions:
                print(f"        Calculating mr for matrix of function: '{name}'")
                try:
                    _, mixing_rate = getattr(module, name)(a, v_)
                    size_results[name].append(mixing_rate)
                except (SolverError, EngineError):
                    size_results[name].append(float('inf'))

        results[str(size)] = size_results
        size += size

    json_string = json.dumps(results, indent=4)
    dir_contents = os.listdir(MIXING_RATE_SAMPLE_ROOT)
    fid = len([*filter(lambda x: "sample" in x, dir_contents)])
    file_path = f"{MIXING_RATE_SAMPLE_ROOT}/sample_{fid + 1}.json"
    with open(file_path, 'w+') as file:
        file.write(json_string)


if __name__ == "__main__":
    samples: int = 30
    max_adj_size: int = 16
    module: Any = "domain.helpers.matrices"
    functions: List[str] = [
        "new_mh_transition_matrix",
        "new_sdp_mh_transition_matrix",
        "new_go_transition_matrix",
        "new_mgo_transition_matrix"
    ]

    try:
        short_opts = "s:a:m:f:"
        long_opts = ["samples=", "adjacency_size=", "module=", "functions="]
        options, args = getopt.getopt(sys.argv[1:], short_opts, long_opts)
        for options, args in options:
            if options in ("-s", "--samples"):
                samples = int(str(args).strip()) or samples
            if options in ("-a", "--adjacency_size"):
                max_adj_size = int(str(args).strip()) or max_adj_size
            if options in ("-m", "--module"):
                module = str(args).strip()
            if options in ("-f", "--functions"):
                function_names = str(args).strip().split(',')

        module = importlib.import_module(module)
        main()
    except getopt.GetoptError:
        sys.exit("Usage: python mixing_rate_sampler.py -s 1000 -f a_matrix_generator")
    except ValueError:
        sys.exit("Execution arguments should have the following data types:\n"
                 "  --samples -s (int)\n"
                 "  --adjacency_size -a (int)\n"
                 "  --module -m (str)\n"
                 "  --functions -f (comma seperated list of str)\n")
    except (ModuleNotFoundError, ImportError):
        sys.exit(f"Module '{module}' does not exist or can not be imported.")
    except AttributeError:
        sys.exit(f"At least a function does not exist in module '{module}'.")
