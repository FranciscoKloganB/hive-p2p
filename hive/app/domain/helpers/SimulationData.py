import numpy as np
import pandas as pd

from typing import Union, List
from globals.globals import MIN_CONVERGENCE_THRESHOLD, R_TOL, A_TOL, MAX_EPOCHS, MAX_EPOCHS_PLUS


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
    :ivar List[int] disconnected_workers: Used to calculate average failures per epoch and cumsum-average failures per epoch
    :ivar List[int] lost_parts: Used to calculate average lost parts per epoch and cumsum-average lost parts per epoch
    :ivar List[int] moved_parts: Used to calculate average parts moved per epoch and cumsum-average parts moved per epoch
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
        self.msg = []  # gathered
        self.disconnected_workers: List[int] = [0] * MAX_EPOCHS  # gathered
        self.lost_parts: List[int] = [0] * MAX_EPOCHS_PLUS  # gathered
        self.hive_status_before_maintenance: List[str] = [""] * MAX_EPOCHS  # gathered
        self.hive_size_before_maintenance: List[int] = [0] * MAX_EPOCHS  # gathered
        self.hive_size_after_maintenance: List[int] = [0] * MAX_EPOCHS  # gathered
        self.delay: List[float] = [0.0] * MAX_EPOCHS_PLUS
        ###############################
        ###############################
        # Updated on Hive.route_part
        self.moved_parts: List[int] = [0] * MAX_EPOCHS  # gathered
        self.corrupted_parts: List[int] = [0] * MAX_EPOCHS  # gathered
        self.lost_messages: List[int] = [0] * MAX_EPOCHS  # gathered
        ###############################
        self.parts_in_hive: List[int] = [0] * MAX_EPOCHS  # gathered
        self.initial_spread = ""

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
    def set_delay_at_index(self, delay: int, calls: int, i: int) -> None:
        """
        :param int delay: the delay sum
        :param int calls: number of times a delay was generated
        :param int i: index of epoch i in SimulationData.delay list
        """
        self.delay[i-1] = 0 if calls == 0 else delay / calls

    def set_disconnected_and_losses(self, disconnected=0, lost=0, i=0):
        """
        Delegates to Hive.set_moved_parts_at_index, Hive.set_failed_workers_at_index, Hive.set_lost_parts_at_index
        """
        self.set_disconnected_workers_at_index(disconnected, i)
        self.set_lost_parts_at_index(lost, i)

    def set_moved_parts_at_index(self, n: int, i: int) -> None:
        """
        :param int n: the quantity of parts moved at epoch i
        :param int i: index of epoch i in SimulationData.moved_parts list
        """
        self.moved_parts[i-1] += n

    def set_parts_at_index(self, n: int, i: int) -> None:
        """
        :param int n: the quantity of parts moved at epoch i
        :param int i: index of epoch i in SimulationData.parts_in_hive list
        """
        self.parts_in_hive[i-1] += n

    def set_disconnected_workers_at_index(self, n: int, i: int) -> None:
        """
        :param int n: the quantity of disconnected workers at epoch i
        :param int i: index of epoch i in SimulationData.disconnected_workers list
        """
        self.disconnected_workers[i-1] += n

    def set_lost_parts_at_index(self, n: int, i: int) -> None:
        """
        :param int n: the quantity of lost parts at epoch i
        :param int i: index of epoch i in SimulationData.lost_parts list
        """
        self.lost_parts[i-1] += n

    def set_lost_messages_at_index(self, n: int, i: int) -> None:
        """
        :param int n: the quantity of loss messages at epoch i
        :param int i: index of epoch i in SimulationData.lost_messages list
        """
        self.lost_messages[i-1] += n

    def set_corrupt_files_at_index(self, n: int, i: int) -> None:
        """
        :param int n: the quantity of corrupted parts at epoch i
        :param int i: index of epoch i in SimulationData.corrupted_parts list
        """
        self.corrupted_parts[i-1] += n

    def set_fail(self, i: int, msg: str = "") -> None:
        """
        Records the epoch at which the Hive terminated, should only be called if it finished early.
        Default, Hive.terminated = MAX_EPOCHS and Hive.successfull = True.
        :param int i: epoch at which Hive terminated
        :param str msg: a message
        """
        self.terminated = i
        self.successfull = False
        self.msg.append(msg)

    def set_membership_maintenace_at_index(self, status: str, size_before: int, size_after: int, i: int) -> None:
        """
        :param string status: status of the hive before maintenance
        :param int size_before: size of the hive before maintenance
        :param int size_after: size of the hive after maintenace
        :param int i: index of epoch i in SimulationData.delay list
        """
        self.hive_status_before_maintenance[i-1] = status
        self.hive_size_before_maintenance[i-1] = size_before
        self.hive_size_after_maintenance[i-1] = size_after
    # endregion

