import os
import pandas as pd
import numpy as np

from math import ceil
from typing import Any, List
from globals.globals import OUTFILE_ROOT, DEBUG, R_TOL
from domain.helpers.ConvergenceData import ConvergenceData
from tabulate import tabulate


class FileData:
    """
    Helper class for domain.Hivemind to keep track of how many parts exist of a file, the number of file parts expected
    to be within the long-term highest density node among other information.
    :ivar str name: the name of the original file
    :ivar int parts_count: the total number of parts the original file was split into, excluding replicas
    :ivar pd.DataFrame desired_distribution: file density distribution hive members must achieve with independent realizations for ideal persistence of the file
    :ivar pd.DataFrame current_distribution: tracks current of file distribution, updated at each epoch
    :ivar ConvergenceData convergence_data: instance object with general information w.r.t. the simulation
    """

    # region Class Variables, Instance Variables and Constructors
    def __init__(self, name: str = "", parts_count: int = 0, cdv: pd.DataFrame = None, convergence_data: ConvergenceData = None):
        self.name: str = name
        self.parts_count: int = parts_count
        self.current_distribution: pd.DataFrame = cdv
        self.convergence_data: ConvergenceData = convergence_data
        self.out_file: Any = open(os.path.join(OUTFILE_ROOT, self.name + ".out"), "w+")
    # endregion

    # region Instance Methods
    def reset_distribution_data(self, labels: List[str], desired_distribution: List[float]) -> None:
        """
        Updates the FileData instance desired_distribution field with the new labeled probabilities and sets current
        distribution to a labeled zero vector
        :param List[str] labels: name of the workers that belong to this file's hive
        :param List[float] desired_distribution: list of probabilities
        """
        # update desired_distribution and reset FileData fields
        self.desired_distribution = pd.DataFrame(desired_distribution, index=labels)
        self.current_distribution = pd.DataFrame([0] * len(desired_distribution), index=labels)

    def reset_convergence_data(self) -> None:
        """
        Resets the FileData instance field convergence_data by delegation to ConvergenceData instance method
        """
        self.convergence_data.save_sets_and_reset()

    def equal_distributions(self) -> bool:
        """
        Delegates distribution comparison to ConvergenceData.equal_distributions static method
        """
        normalized_cdv = self.current_distribution.divide(self.parts_count)
        if DEBUG:
            self.fwrite("Desired Distribution:\n{}\nCurrent Distribution:\n{}\n".format(
                tabulate(self.desired_distribution, headers='keys', tablefmt='psql'),
                tabulate(normalized_cdv, headers='keys', tablefmt='psql')
            ))
        return np.allclose(self.desired_distribution, normalized_cdv, rtol=R_TOL, atol=(1 / self.parts_count))
        # return ConvergenceData.equal_distributions(self.desired_distribution, normalized_cdv)

    def get_failure_threshold(self) -> int:
        """
        Calculates the maximum amount of files of a given file that can be lost at any given time
        :returns int: the failure threshold for the file being represented by the FileData instance
        """
        highest_density_node_label = self.desired_distribution.idxmax().values[0]  # index/label of highval
        highest_density_node_density = self.desired_distribution.at[highest_density_node_label, 0]  # highval
        return self.parts_count - ceil(self.parts_count * highest_density_node_density)
    # endregion

    # region File I/O
    def fwrite(self, string: str) -> None:
        """
        Writes to the out_file referenced by the FileData's instance out_file field in append mode
        :param str string: a message to write to file
        """
        print(string)
        self.out_file.write(string + "\n")

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
