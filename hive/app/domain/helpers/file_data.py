import json
import os

import numpy as np
import pandas as pd

from pathlib import Path
from tabulate import tabulate
from typing import Any, Union
from globals.globals import OUTFILE_ROOT, DEBUG, R_TOL
from domain.helpers.simulation_data import SimulationData


class FileData:
    """
    Helper class for domain.Hivemind to keep track of how many parts exist of a file, the number of file parts expected
    to be within the long-term highest density node among other information.
    :ivar str name: the name of the original file
    :ivar int parts_count: the total number of parts the original file was split into, excluding replicas
    :ivar pd.DataFrame desired_distribution: file density distribution hive members must achieve with independent realizations for ideal persistence of the file
    :ivar pd.DataFrame current_distribution: tracks current of file distribution, updated at each epoch
    :ivar ConvergenceData simulation_data: instance object with general information w.r.t. the simulation
    """

    # region Class Variables, Instance Variables and Constructors
    def __init__(self, name: str, sim_number: int = 0):
        """
        :param str name: name of the file referenced by this data class instance
        :param int sim_number: optional value that can be passed to FileData to generate different .out names
        """
        self.name: str = Path(name).resolve().stem
        self.desired_distribution: Union[None, pd.DataFrame] = None
        self.current_distribution: Union[None, pd.DataFrame] = None
        self.simulation_data: SimulationData = SimulationData()
        self.out_file: Any = open(os.path.join(OUTFILE_ROOT, "{}_{}{}".format(self.name, sim_number, ".out" if DEBUG else ".json")), "w+")
    # endregion

    # region Instance Methods
    def equal_distributions(self, parts_in_hive: int) -> bool:
        """
        Delegates distribution comparison to ConvergenceData.equal_distributions static method
        """
        if parts_in_hive == 0:
            return False

        normalized_cdv = self.current_distribution.divide(parts_in_hive)
        if DEBUG:
            self.fwrite("Desired Distribution:\n{}\nCurrent Distribution:\n{}\n".format(
                tabulate(self.desired_distribution, headers='keys', tablefmt='psql'), tabulate(normalized_cdv, headers='keys', tablefmt='psql')
            ))
        return np.allclose(self.desired_distribution, normalized_cdv, rtol=R_TOL, atol=(1 / parts_in_hive))
    # endregion

    # region File I/O
    def fwrite(self, string: str) -> None:
        """
        Writes to the out_file referenced by the FileData's instance out_file field in append mode
        :param str string: a message to write to file
        """
        self.out_file.write(string + "\n")

    def jwrite(self, data: SimulationData, epoch: int):
        if not data.msg:
            data.msg.append("completed simulation successfully")
        if DEBUG:
            [print("* {};".format(reason)) for reason in data.msg]
        data.disconnected_workers = data.disconnected_workers[:epoch]
        data.delay = data.delay[:epoch]
        data.lost_parts = data.lost_parts[:epoch]
        data.hive_status_before_maintenance = data.hive_status_before_maintenance[:epoch]
        data.hive_size_before_maintenance = data.hive_size_before_maintenance[:epoch]
        data.hive_size_after_maintenance = data.hive_size_after_maintenance[:epoch]
        data.delay = data.delay[:epoch]
        json_string = json.dumps(data.__dict__, indent=4, sort_keys=True, ensure_ascii=False)
        self.fwrite(json_string)

    def fclose(self, string: str = None) -> None:
        """
        Closes the out_file referenced by the FileData's instance out_file field
        :param str string: if filled, a message is written in append mode before closing the out_file
        """
        if string:
            self.fwrite(string)
        self.out_file.close()
    # endregion

    # region Overrides
    def __hash__(self):
        # allows a worker object to be used as a dictionary key
        return hash(str(self.name))

    def __eq__(self, other):
        if not isinstance(other, FileData):
            return False
        return self.name == other.name

    def __ne__(self, other):
        return not(self == other)
    # endregion

    # region Helpers
    def reset_convergence_data(self) -> None:
        """
        Resets the FileData instance field simulation_data by delegation to ConvergenceData instance method
        """
        self.simulation_data.save_sets_and_reset()
    # endregion
