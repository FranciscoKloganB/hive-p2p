"""Excludes all  <topologies, equilibrium> pairs in the ``scenarios.json`` file
that are not synthetizable by our implementation of
:py:meth:`Metropolis Hastings
<app.domain.helpers.matrices._metropolis_hastings>`. Such JSON file is created
using the script :py:mod:`sample_scenario_generator`.

To execute this file run the following command::

    $ python sample_scenario_generator.py

Note:
    This script expects to fix a file named "scenarios.json" under the
    :py:const:`~app.environment_settings.RESOURCES_ROOT` directory. If you wish
    to modify this behavior you need to customize the script to accept one
    additional argument which indicates the name of the file to be fixed.
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import environment_settings as es
import domain.helpers.matrices as mm


def __validate_mc__(m: pd.DataFrame, v_: pd.DataFrame) -> bool:
    """Asserts if the inputed Markov Matrix that converges to the desired equilibrium."""
    t_pow = np.linalg.matrix_power(m.to_numpy(), 4096)
    column_count = t_pow.shape[1]
    for j in range(column_count):
        test_target = t_pow[:, j]  # gets array column j
        if not np.allclose(
                test_target, v_[0].values, atol=1e-02):
            return False
    return True


def __select_fastest_topology__(a: np.ndarray, v_: np.ndarray) -> np.ndarray:
    """Emulates Swarm Guidance Clusters' fastest topology selection for MH algorithms."""
    fastest_matrix, _ = mm.new_mh_transition_matrix(a, v_)
    size = fastest_matrix.shape[0]
    for j in range(size):
        fastest_matrix[:, j] = np.absolute(fastest_matrix[:, j])
        fastest_matrix[:, j] /= fastest_matrix[:, j].sum()
    return fastest_matrix


if __name__ == "__main__":
    scenarios_path = os.path.join(es.RESOURCES_ROOT, "scenarios.json")
    if not os.path.exists(scenarios_path):
        sys.exit("Scenarios file does not exist.")

    with open(scenarios_path, "r+") as f:
        scenarios_dict = json.load(f)
        for sk in scenarios_dict:
            invalid_pair_indexes = []
            samples_dict = scenarios_dict[sk]
            matrices = samples_dict["matrices"]
            vectors = samples_dict["vectors"]
            for i in range(len(vectors)):
                # print(f"Checking scenario #{i + 1}...")
                a = np.asarray(matrices[i])
                v = np.asarray(vectors[i])
                m = __select_fastest_topology__(a, v)
                is_valid_mc = __validate_mc__(pd.DataFrame(m), pd.DataFrame(v))
                if not is_valid_mc:
                    invalid_pair_indexes.append(i)
            print(f"Found {len(invalid_pair_indexes)} invalid scenarios for size {sk}.")
            invalid_pair_indexes.sort(reverse=True)
            for i in invalid_pair_indexes:
                print(f"Removing scenario entry #{i}")
                del matrices[i]
                del vectors[i]
        f.seek(0)  # Place cursor at start of file
        f.write(json.dumps(
            scenarios_dict, indent=4, sort_keys=True, ensure_ascii=False))
        f.truncate()  # Remove left characters.
        print("Operation complete.")
