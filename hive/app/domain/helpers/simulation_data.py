import numpy as np
import pandas as pd

from typing import Union, List
from globals.globals import MIN_CONVERGENCE_THRESHOLD, R_TOL, A_TOL, MAX_EPOCHS


class SimulationData:
    # region docstrings
    """
    Holds data that helps an domain.Hivemind keep track of converge in a simulation
    :ivar int cswc: indicates how many consecutive steps a file has in convergence
    :ivar int largest_convergence_window: indicates the biggest set of consecutive steps throughout the simulation for a file
    :ivar int terminated: indicates the epoch at which the Hive terminated
    :ivar bool successfull: indicates if Hive survived the entire simulation
    :ivar List[int] convergence_set: current consecutive set of stages in which a file has seen convergence
    :ivar List[List[int]] convergence_sets: Set of all convergence sets found for this file during simulation
    :ivar List[int] disconnected_workers_per_epoch: Used to calculate average failures per epoch and cumsum-average failures per epoch
    :ivar List[int] lost_parts_per_epoch: Used to calculate average lost parts per epoch and cumsum-average lost parts per epoch
    :ivar List[int] moved_parts_per_epoch: Used to calculate average parts moved per epoch and cumsum-average parts moved per epoch
    """
    # endregion

    # region Class Variables, Instance Variables and Constructors
    def __init__(self):
        self.cswc: int = 0
        self.largest_convergence_window: int = 0
        self.convergence_set: List[int] = []
        self.convergence_sets: List[List[int]] = []
        ###############################
        # Updated on Hive.execute_epoch
        self.terminated: int = MAX_EPOCHS  # gathered
        self.successfull: bool = True  # gathered
        self.msg = "completed simulation successfully"  # gathered
        self.disconnected_workers_per_epoch: List[int] = [0] * MAX_EPOCHS  # gathered
        self.lost_parts_per_epoch: List[int] = [0] * MAX_EPOCHS  # gathered
        ###############################
        ###############################
        # Updated on Hive.route_part
        self.moved_parts_per_epoch: List[int] = [0] * MAX_EPOCHS
        self.corrupted_parts_per_epoch: List[int] = [0] * MAX_EPOCHS
        self.lost_messages_per_epoch: List[int] = [0] * MAX_EPOCHS
        ###############################

    # endregion

    # region Instance Methods
    def cswc_increment(self, increment: int = 1) -> None:
        """
        Increases the counter that which registers how many consecutive stages had convergence
        :param increment: value to increment; default is 1
        """
        self.cswc += increment

    def try_set_largest_convergence_set(self) -> None:
        """
        Verifies if the current convergence set is the largest of all seen so far, if it is, updates the ConvergenceData
        instance field largest_convergence_window to be the length of the convergence_set
        """
        set_len = len(self.convergence_set)
        if set_len > self.largest_convergence_window:
            self.largest_convergence_window = set_len

    def try_append_to_convergence_set(self, epoch: int) -> None:
        """
        Checks if the counter for consecutive epoch convergence is bigger than the minimum threshold for verified
        convergence and if it is, appends the inputted epoch to the current convergence set
        :param epoch:
        """
        if self.cswc >= MIN_CONVERGENCE_THRESHOLD:
            self.convergence_set.append(epoch)

    def save_sets_and_reset(self) -> None:
        """
        Appends the current convergence set to the list of convergence sets, verifies if the saved set is the largest
        seen so far and resets consecutive stages with convergence counter to zero.
        """
        self.cswc = 0
        if self.convergence_set:  # code below only executes if convergence_set isn't empty, i.e.: convergence_set != []
            self.try_set_largest_convergence_set()
            self.convergence_sets.append(self.convergence_set)
            self.convergence_set = []
    # endregion

    # region Static Methods
    @staticmethod
    def equal_distributions(one: pd.DataFrame, another: pd.DataFrame) -> bool:
        """
        :param pd.DataFrame one: labeled distribution
        :param pd.DataFrame another: another labeled distribution
        :returns bool: True if the elements at each index of both distributions are close enough, or False otherwise
        """
        if len(one) != len(another):
            return False
        else:
            return np.allclose(one.sort_index(), another.sort_index(), rtol=R_TOL, atol=A_TOL)

    @staticmethod
    def recursive_len(item):
        if type(item) == list:
            return sum(SimulationData.recursive_len(sub_item) for sub_item in item)
        else:
            return 1
    # endregion

    # region Overrides
    def __str__(self):
        return "time in convergence: " + str(SimulationData.recursive_len(self.convergence_sets)) + \
               "\nlargest_convergence_window: " + str(self.largest_convergence_window) + \
               "\nconvergence_sets:\n " + str(self.convergence_sets)

    def __repr__(self):
        return str(self.__dict__)
    # endregion

    # region Helpers
    def set_epoch_data(self, disconnected=0, lost=0, epoch=0):
        """
        Delegates to Hive.set_moved_parts_at_index, Hive.set_failed_workers_at_index, Hive.set_lost_parts_at_index
        """
        self.set_failed_workers_at_index(disconnected, epoch)
        self.set_lost_parts_at_index(lost, epoch)

    def set_moved_parts_at_index(self, n: int, i: int) -> None:
        """
        :param int n: the quantity of parts moved at epoch i
        :param int i: index of epoch i in SimulationData.moved_parts_per_epoch list
        """
        self.moved_parts_per_epoch[i] += n

    def set_failed_workers_at_index(self, n: int, i: int) -> None:
        """
        :param int n: the quantity of disconnected workers at epoch i
        :param int i: index of epoch i in SimulationData.disconnected_workers_per_epoch list
        """
        self.disconnected_workers_per_epoch[i] += n

    def set_lost_parts_at_index(self, n: int, i: int) -> None:
        """
        :param int n: the quantity of lost parts at epoch i
        :param int i: index of epoch i in SimulationData.lost_parts_per_epoch list
        """
        self.lost_parts_per_epoch[i] += n

    def set_fail(self, i: int, msg: str = "") -> bool:
        """
        Records the epoch at which the Hive terminated, should only be called if it finished early.
        Default, Hive.terminated = MAX_EPOCHS and Hive.successfull = True.
        :param int i: epoch at which Hive terminated
        :param str msg: a message
        :returns bool: usually returns False, only returns True when param i, representing epoch is qual to MAX_EPOCHS
        """
        if i == MAX_EPOCHS:
            return True

        self.terminated = i
        self.successfull = False
        self.msg = msg
        return False
    # endregion
