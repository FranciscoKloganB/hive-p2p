import numpy as np
import pandas as pd

from typing import Union, List, Any
from globals.globals import MIN_CONVERGENCE_THRESHOLD, R_TOL, A_TOL, MAX_EPOCHS, MAX_EPOCHS_PLUS


class SimulationData:
    """Logging class that registers simulation state per epoch basis.

    Notes:
        Most attributes of this class are not documented in docstrings,
        but they are straight forward to understand. They are mostly lists of
        length :py:const:`MAX_EPOCHS <globals.globals.MAX_EPOCHS>` that
        contain data concerning the current state of simulation at the
        respective epoch times. For example, :py:attr:`~lost_parts` keeps
        a integers that represent how many file blocks were lost at each
        epoch of the simulation and :py:attr:`~moved_parts` registers
        the number of file block messages that traveled the network at the
        same respective epoch. If you wish to monitor any property (or not) of
        the simulation you should modify this class.

    Attributes:
        cswc (int):
            Indicates how many consecutive steps a file as been in
            convergence. Once convergence is not verified by
            :py:meth:`equal_distributions() <domain.Hive.Hive.equal_distributions>`
            this attribute is reseted to zero.
        largest_convergence_window (int):
            Stores the largest convergence window that occurred throughout
            the simulation, i.e., it stores the highest verified
            :py:attr:`~cswc`.
        convergence_set (list of ints):
            Set of consecutive epochs in which convergence was verified.
            This list only stores the most up to date convergence set and like
            :py:attr:`~cswc` is cleared once convergence is not verified,
            after being appended to :py:attr:`~convergence_sets`.
        convergence_sets (list of lists of ints):
            Stores all previous convergence sets. See :py:attr:`~convergence_set`.
        terminated (int):
            Indicates the epoch at which the simulation was terminated.
        successfull (bool):
            When the simulation is terminated this value is set to True if
            no errors or failures occurred, i.e., if the simulation managed
            to persist the file throughout
            :py:const:`MAX_EPOCHS <globals.globals.MAX_EPOCHS>` time steps.
        messages (list of str):
            Set of at least one error message that led to the failure
            of the simulation or one success message, at termination epoch
            (:py:attr:`~terminated`)
    """
    # endregion

    # region Class Variables, Instance Variables and Constructors
    def __init__(self) -> None:
        """Instanciates a SimulationData object for simulation event logging."""
        ###############################
        # Do not alter these
        self.cswc: int = 0
        self.largest_convergence_window: int = 0
        self.convergence_set: List[int] = []
        self.convergence_sets: List[List[int]] = []
        self.terminated: int = MAX_EPOCHS
        self.successfull: bool = True
        self.messages = []
        ###############################
        ###############################
        # Alter these
        self.disconnected_workers: List[int] = [0] * MAX_EPOCHS
        self.lost_parts: List[int] = [0] * MAX_EPOCHS_PLUS
        self.hive_status_before_maintenance: List[str] = [""] * MAX_EPOCHS
        self.hive_size_before_maintenance: List[int] = [0] * MAX_EPOCHS
        self.hive_size_after_maintenance: List[int] = [0] * MAX_EPOCHS
        self.delay: List[float] = [0.0] * MAX_EPOCHS_PLUS
        self.moved_parts: List[int] = [0] * MAX_EPOCHS
        self.corrupted_parts: List[int] = [0] * MAX_EPOCHS
        self.lost_messages: List[int] = [0] * MAX_EPOCHS
        self.parts_in_hive: List[int] = [0] * MAX_EPOCHS
        self.initial_spread = ""
        ###############################

    # endregion

    # region Instance Methods

    def register_convergence(self, epoch: int) -> None:
        """Increments :py:attr:`~cswc` by one and tries to update the :py:attr:`~convergence_set`

        Checks if the counter for consecutive epoch convergence is bigger
        than the minimum threshold for verified convergence (see
        :py:const:`MIN_CONVERGENCE_THRESHOLD <globals.globals.MIN_CONVERGENCE_THRESHOLD>`
        and if it is, it marks the epoch as part of the current
        :py:attr:`~convergence_set`.

        Args:
            epoch:
                The simulation's current epoch.
        """
        self.cswc += 1
        if self.cswc >= MIN_CONVERGENCE_THRESHOLD:
            self.convergence_set.append(epoch)

    def save_sets_and_reset(self) -> None:
        """Resets all convergence variables

        Tries to update :py:attr:`~largest_convergence_window` and
        :py:attr:`~convergence_sets` when :py:attr:`~convergence_set`
        is not an empty list.
        """
        self.cswc = 0
        if len(self.convergence_set) > 0:
            # check if current convergence set is the biggest so far
            set_len = len(self.convergence_set)
            if set_len > self.largest_convergence_window:
                self.largest_convergence_window = set_len
            self.convergence_sets.append(self.convergence_set)
            self.convergence_set = []

    def _recursive_len(self, item: Any) -> int:
        """Recusively sums the length of all lists in :py:attr:`~convergence_sets`.

        Args:
            item: A sub list of :py:attr:`~convergence_sets` that needs that
            as not yet been counted.

        Returns:
            The number of epochs that were registered at the inputed sub list.
        """
        if type(item) == list:
            return sum(self._recursive_len(sub_item) for sub_item in item)
        return 1

    # endregion

    # region Overrides

    def __str__(self):
        return "time in convergence: " + str(self._recursive_len(self.convergence_sets)) + \
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
        self.messages.append(msg)

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

