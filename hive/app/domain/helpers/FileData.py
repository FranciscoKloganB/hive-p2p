import os
import pandas as pd

from math import ceil
from typing import Dict, Any, List

from pandas import DataFrame

from globals.globals import OUTFILE_ROOT
from domain.helpers.ConvergenceData import ConvergenceData


class FileData:
    """
    Helper class for domain.Hivemind to keep track of how many parts exist of a file, the number of file parts expected
    to be within the long-term highest density node among other information.
    :ivar str file_name: the name of the file
    :ivar int parts_count: how many parts exist for the
    :ivar str highest_density_node_label: label of the highest density node
    :ivar float highest_density_node_density: file density for the highest density node
    :ivar pd.DataFrame desired_distribution: file distribution hive members must achieve with independent realizations
    :ivar pd.DataFrame current_distribution: tracks current of file distribution, updated at each discrete time step
    :ivar ConvergenceData convergence_data: instance object with general information perteining the simulation
    :ivar pd.DataFrame adjacency_matrix: current hive members' connections
    """
    current_distribution: DataFrame

    # region class variables, instance variables and constructors
    def __init__(self,
                 file_name: str = "",
                 parts_count: int = 0,
                 node_name: str = "",
                 density: float = 0.0,
                 ddv: pd.DataFrame = None,
                 cdv: pd.DataFrame = None,
                 convergence_data: ConvergenceData = None,
                 adj_matrix: pd.DataFrame = None):
        self.file_name: str = file_name
        self.parts_count: int = parts_count
        self.highest_density_node_label: str = node_name
        self.highest_density_node_density: float = density
        self.desired_distribution: pd.DataFrame = ddv
        self.current_distribution: pd.DataFrame = cdv
        self.convergence_data: ConvergenceData = convergence_data
        self.adjacency_matrix: pd.DataFrame = adj_matrix
        self.out_file: Any = open(os.path.join(OUTFILE_ROOT, self.file_name + ".out"), "w+")
    # endregion

    # region instance methods
    def commit_replacement(self, replacement_dict: Dict[str, str]) -> None:
        """
        Replaces names of dead nodes in desired and current distribution data frames and updates remaining structures
        accordingly
        :param Dict[str, str] replacement_dict: dictionary containing a mapping of key names to replace with value names
        """
        self.replace_adjacency_node(replacement_dict)
        self.replace_distribution_node(replacement_dict)
        self.reset_density_data()
        self.reset_convergence_data()

    def reset_adjacency_matrix(self, labels: List[str], adjacency_matrix: List[List[int]]) -> None:
        """
        Updates the FileData instance adjacency_matrix field with new labeled adjacency matrix
        :param List[str] labels: name of the workers that belong to this file's hive
        :param List[List[int]] adjacency_matrix: adjacency matrix representing connections between various states
        """
        self.adjacency_matrix = pd.DataFrame(adjacency_matrix, index=labels, columns=labels)

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

    def reset_density_data(self) -> None:
        """
        Updates FileData instance highest_density_node_label field to be the name of the node with highest file density
        and stores that density value on highest_density_node_density field
        """
        self.highest_density_node_label = self.desired_distribution.idxmax().values[0]  # index/label of highval
        self.highest_density_node_density = self.desired_distribution.at[self.highest_density_node_label, 0]  # highval

    def reset_convergence_data(self) -> None:
        """
        Resets the FileData instance field convergence_data by delegation to ConvergenceData instance method
        """
        self.convergence_data.save_sets_and_reset()

    def replace_adjacency_node(self, replacement_dict: Dict[str, str]) -> None:
        """
        Replaces a row and a column label name with a new label on the FileData instance adjacency_matrix field
        :param Dict[str, str] replacement_dict: old worker name, new worker name)
        """
        self.adjacency_matrix.rename(index=replacement_dict, columns=replacement_dict, inplace=True)

    def replace_distribution_node(self, replacement_dict: Dict[str, str]) -> None:
        """
        Replaces a row label with a new label, on the FileData instance desired and current distribution fields
        :param Dict[str, str] replacement_dict: old worker name, new worker name)
        """
        self.desired_distribution.rename(index=replacement_dict, inplace=True)
        self.current_distribution.rename(index=replacement_dict, inplace=True)

    def equal_distributions(self) -> bool:
        """
        Delegates distribution comparison to ConvergenceData.equal_distributions static method
        """
        return ConvergenceData.equal_distributions(self.desired_distribution, self.current_distribution.divide(self.parts_count))

    def get_failure_threshold(self) -> int:
        """
        Calculates the maximum amount of files of a given file that can be lost at any given time
        :returns int: the failure threshold for the file being represented by the FileData instance
        """
        return self.parts_count - ceil(self.parts_count * self.highest_density_node_density)
    # endregion

    # region file I/O
    def fwrite(self, string: str) -> None:
        """
        Writes to the out_file referenced by the FileData's instance out_file field in append mode
        :param str string: a message to write to file
        """
        print(string)
        self.out_file.write(string + "\n")

    def fclose(self) -> None:
        """
        Closes the out_file referenced by the FileData's instance out_file field
        """
        self.out_file.close()
    # endregion
