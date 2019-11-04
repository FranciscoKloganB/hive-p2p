import os
import pandas as pd

from math import ceil
from globals.globals import OUTFILE_ROOT
from domain.helpers.ConvergenceData import ConvergenceData as cD


class FileData:
    # region docstrings
    """
    Helper class for domain.Hivemind to keep track of how many parts exist of a file, the number of file parts expected
    to be within the long-term highest density node among other information.
    :ivar file_name: the name of the file
    :type str
    :ivar parts_count: how many parts exist for the
    :type int
    :ivar highest_density_node_label: label of the highest density node
    :type str
    :ivar highest_density_node_density: file density for the highest density node
    :type float
    :ivar desired_distribution: desired distribution that sharers of this file must achieve together but independently
    :type pandas.Dataframe, column vector with labels
    :ivar current_distribution: keeps track of file current distribution, at each discrete time stage
    :type dict<str, list<float>>
    :ivar convergence_data: instance object with general information perteining the simulation
    :type domain.helpers.ConvergenceData
    """
    # endregion

    # region class variables, instance variables and constructors
    def __init__(self,
                 file_name="",
                 parts_count=0,
                 node_name="",
                 density=0.0,
                 ddv=None,
                 cdv=None,
                 convergence_data=None,
                 adj_matrix=None):
        self.file_name = file_name
        self.parts_count = parts_count
        self.highest_density_node_label = node_name
        self.highest_density_node_density = density
        self.desired_distribution = ddv
        self.current_distribution = cdv
        self.convergence_data = convergence_data
        self.adjacency_matrix = adj_matrix
        self.out_file = open(os.path.join(OUTFILE_ROOT, self.file_name + ".out"), "w+")
    # endregion

    # region instance methods
    def reset_adjacency_matrix(self, labels, adjacency_matrix):
        """
        :param labels: name of the workers that belong to this file's hive
        :type list<str>
        :param adjacency_matrix: adjacency matrix representing connections between various states
        :type list<list<int>>
        """
        self.adjacency_matrix = pd.DataFrame(adjacency_matrix, index=labels, columns=labels)

    def reset_distribution_data(self, labels, desired_distribution):
        """
        :param labels: name of the workers that belong to this file's hive
        :type list<str>
        :param desired_distribution: list of probabilities
        :type list<float>
        """
        # update desired_distribution and reset FileData fields
        self.desired_distribution = pd.DataFrame(desired_distribution, index=labels)
        self.current_distribution = pd.DataFrame([0] * len(desired_distribution), index=labels)

    def reset_density_data(self):
        self.highest_density_node_label = self.desired_distribution.idxmax().values[0]  # index/label of highval
        self.highest_density_node_density = self.desired_distribution.at[self.highest_density_node_label, 0]  # highval

    def reset_convergence_data(self):
        self.convergence_data.save_sets_and_reset()

    def replace_distribution_node(self, replacement_dict):
        self.desired_distribution.rename(index=replacement_dict, inplace=True)
        self.current_distribution.rename(index=replacement_dict, inplace=True)

    def equal_distributions(self):
        cD.equal_distributions(self.desired_distribution, self.current_distribution)

    def get_failure_threshold(self):
        """
        :returns: the failure threshold for the given file
        :type int
        """
        return self.parts_count - ceil(self.parts_count * self.highest_density_node_density)
    # endregion

    # region file I/O
    def fwrite(self, string):
        self.out_file.write(string + "\n")

    def fclose(self):
        self.out_file.close()
    # endregion
