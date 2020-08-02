from typing import List, Any

from globals.globals import MIN_CONVERGENCE_THRESHOLD
from hive_simulation import MAX_EPOCHS, MAX_EPOCHS_PLUS


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
                The A simulation epoch index.
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
        set_len = len(self.convergence_set)
        if set_len > 0:
            self.convergence_sets.append(self.convergence_set)
            self.convergence_set = []
            if set_len > self.largest_convergence_window:
                self.largest_convergence_window = set_len
        self.cswc = 0

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

    def set_delay_at_index(self, delay: int, calls: int, epoch: int) -> None:
        """Logs the expected delay at epoch at an epoch.

        Args:
            delay:
                The delay sum.
            calls:
                Number of times a delay was generated.
            epoch:
                A simulation epoch index.
        """
        self.delay[epoch-1] = 0 if calls == 0 else delay / calls

    def set_moved_parts_at_index(self, n: int, epoch: int) -> None:
        """Logs the amount of moved file blocks moved at an epoch.

        Args:
            n:
                Number of parts moved at epoch.
            epoch:
                A simulation epoch index.
        """
        self.moved_parts[epoch-1] += n

    def set_parts_at_index(self, n: int, epoch: int) -> None:
        """Logs the amount of existing file blocks in the simulation environment at an epoch.

        Args:
            n:
                Number of file blocks in the system.
            epoch:
                A simulation epoch index.
        """
        self.parts_in_hive[epoch-1] += n

    def set_disconnected_workers_at_index(self, n: int, epoch: int) -> None:
        """Logs the amount of disconnected workers at an epoch.

        Args:
            n:
                Number of disconnected workers in the system.
            epoch:
                A simulation epoch index.
        """
        self.disconnected_workers[epoch-1] += n

    def set_lost_parts_at_index(self, n: int, epoch: int) -> None:
        """Logs the amount of permanently lost file block replicas at an epoch.

        Args:
            n:
                Number of replicas that were lost.
            epoch:
                A simulation epoch index.
        """
        self.lost_parts[epoch-1] += n

    def set_lost_messages_at_index(self, n: int, epoch: int) -> None:
        """Logs the amount of failed message transmissions at an epoch.

        Args:
            n:
                Number of lost messages.
            epoch:
                A simulation epoch index.
        """
        self.lost_messages[epoch-1] += n

    def set_corrupt_files_at_index(self, n: int, epoch: int) -> None:
        """Logs the amount of corrupted file block replicas at an epoch.

        Args:
            n:
                Number of corrupted blocks
            epoch:
                A simulation epoch index.
        """
        self.corrupted_parts[epoch-1] += n

    def set_fail(self, epoch: int, message: str = "") -> None:
        """Logs the epoch at which a simulation terminated due to a failure.

        Note:
            This method should only be called when simulation terminates due
            to a failure such as a the loss of all replicas of a file block
            or the simultaneous disconnection of all network nodes in the hive.

        Args:
            message:
                optional; A log error message (default is blank)
            epoch:
                A simulation epoch at which termination occurred.
        """
        self.terminated = epoch
        self.successfull = False
        self.messages.append(message)

    def set_membership_maintenace_at_index(self,
                                           status: str,
                                           size_before: int,
                                           size_after: int,
                                           epoch: int) -> None:
        """Logs hive membership status and size at an epoch.

        Args:
            status:
                A string that describes the status of the hive after
                maintenance.
            size_before:
                The number of network nodes in the hive before maintenance.
            size_after:
                The number of network nodes in the hive after maintenance.
            epoch:
                A simulation epoch at which termination occurred.
        """
        self.hive_status_before_maintenance[epoch-1] = status
        self.hive_size_before_maintenance[epoch-1] = size_before
        self.hive_size_after_maintenance[epoch-1] = size_after

    # endregion

