"""Creates an arbrirary number of symmetric connected topologies and equilibrium
vectors that can be read during simulations for one to one comparison between
algorithms. There is a small chance that generated pairs can not be solved by
heuristic Markov chain generating algorithms such as our implementation of
:py:meth:`Metropolis Hastings
<app.domain.helpers.matrices._metropolis_hastings>`. To ensure that algorithm
can be used over the generated pairs, run :py:mod:`sample_scenario_fixer`,
which removes all invalid entries from the generated json file.

To execute this file run the following command (both arguments are optional)::

    $ python sample_scenario_generator.py --samples=1000 --network_sizes=8,16,32

Note:
    The output of this script is a file named "scenarios.json" under the
    :py:const:`~app.environment_settings.RESOURCES_ROOT` directory. If you wish
    to modify this behavior you need to customize the script to accept one
    additional argument which than saves the file under a different name. You
    also need to ensure that all other uses of "scenarios.json" are changed
    accordingly.
"""

import os
import sys
import ast
import json
import getopt

import environment_settings as es
import domain.helpers.matrices as mm

from typing import Tuple, Any

if __name__ == "__main__":
    network_sizes: Tuple = (8, 16, 32)
    samples: int = 100

    short_opts = "n:s:"
    long_opts = ["network_sizes=", "samples="]

    try:
        options, args = getopt.getopt(sys.argv[1:], short_opts, long_opts)
        for options, args in options:
            if options in ("-n", "--network_size"):
                network_sizes = ast.literal_eval(str(args).strip())
            if options in ("-s", "--samples"):
                samples = int(str(args).strip())
    except getopt.GetoptError:
        sys.exit()
    except ValueError:
        sys.exit("Execution arguments should have the following data types:\n"
                 "  --network_size -n (comma seperated list of int)\n"
                 "  --samples -s (int)\n")

    if not network_sizes or samples < 1:
        sys.exit("Can't proceed with no samples parameter or empty networks.")

    os.makedirs(es.RESOURCES_ROOT, exist_ok=True)

    scenarios = {str(k): {"vectors": [], "matrices": []} for k in network_sizes}
    for k in scenarios:
        network_size = int(k)
        for i in range(samples):
            v_ = mm.new_vector(network_size).tolist()
            m = mm.new_symmetric_connected_matrix(network_size).tolist()
            scenarios[k]["vectors"].append(v_)
            scenarios[k]["matrices"].append(m)

    with open(os.path.join(es.RESOURCES_ROOT, "scenarios.json"), "w") as f:
        json.dump(scenarios, f, indent=4, sort_keys=True, ensure_ascii=False)
