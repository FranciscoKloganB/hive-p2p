"""Module with classes that help to avoid domain class polution by
encapsulating attribute and method behaviour."""
from __future__ import annotations

import json

import domain.cluster_groups as cg
import domain.master_servers as ms

from pathlib import Path
from random import randint
from typing import Any, Dict, IO, List
from utils import convertions, crypto
from environment_settings import *


class FileData:
    """Holds essential simulation data concerning files being persisted.

    FileData is a helper class which has responsabilities such as tracking
    how many parts including blocks exist of the named file and managing
    the persistence of logged simulation data to disk.

    Attributes:
        name (str):
            The name of the original file.
        parts_in_hive (int):
            The number of file parts including blocks that exist for the
            named file that exist in the simulation. Updated every epoch.
        logger (LoggingData):
            Object that stores captured simulation data. Stored data can be
            post-processed using user defined scripts to create items such
            has graphs and figures. See
            :py:class:`~app.domain.helpers.smart_dataclasses.SimulationData`
        out_file (str/bytes/int):
            File output stream to where captured data is written in append mode.
    """

    def __init__(self, name: str, sim_id: int = 0, origin: str = "") -> None:
        """Creates an instance of FileData

        Args:
            name:
                Name of the file to be referenced by the FileData object.
            sim_id:
                optional; Identifier that generates unique output file names,
                thus guaranteeing that different simulation instances do not
                overwrite previous out files.
            origin:
                optional; The name of the simulation file name that started
                the simulation process.
        """
        self.name: str = name
        self.parts_in_hive = 0
        self.logger: LoggingData = LoggingData()
        self.out_file: IO = open(
            os.path.join(
                OUTFILE_ROOT, "{}_{}{}.{}".format(
                    Path(name).resolve().stem,
                    Path(origin).resolve().stem,
                    sim_id,
                    "json")
            ), "w+")

    def fwrite(self, msg: str) -> None:
        """Writes a message to the output file referenced by the FileData object.

        The method fwrite automatically adds a new line to the inputted message.

        Args:
            msg:
                The message to be logged on the output file.
        """
        self.out_file.write(msg + "\n")

    def jwrite(self, hive: cg.Cluster, origin: str, epoch: int) -> None:
        """Writes a JSON string of the LoggingData instance to the output file.

        The logged data is defined by the attributes of the
        :py:class:`LoggingData <app.domain.helpers.smart_dataclasses.LoggingData`
         class.

        Args:
            hive:
                The :py:class:`Cluster <app.domain.cluster_groups.Cluster>` object that manages
                the simulated persistence of the referenced file.
            origin:
                The name of the simulation file that started the simulation
                process.
            epoch:
                The epoch at which the LoggingData was logged into the
                output file.

        """
        sd: LoggingData = self.logger

        sd.save_sets_and_reset()

        if not sd.terminated_messages:
            sd.terminated_messages.append("completed simulation successfully")

        sd.blocks_existing = sd.blocks_existing[:epoch]

        sd.off_node_count = sd.off_node_count[:epoch]
        sd.blocks_lost = sd.blocks_lost[:epoch]

        sd.cluster_status_bm = sd.cluster_status_bm[:epoch]
        sd.cluster_size_bm = sd.cluster_size_bm[:epoch]
        sd.cluster_size_am = sd.cluster_size_am[:epoch]

        sd.delay_replication = sd.delay_replication[:epoch]

        sd.blocks_moved = sd.blocks_moved[:epoch]
        sd.blocks_corrupted = sd.blocks_corrupted[:epoch]
        sd.transmissions_failed = sd.transmissions_failed[:epoch]

        extras: Dict[str, Any] = {
            "simfile_name": origin,
            "hive_id": hive.id,
            "file_name": self.name,
            "read_size": READ_SIZE,
            "critical_size_threshold": hive.critical_size,
            "sufficient_size_threshold": hive.sufficient_size,
            "original_hive_size": hive.original_size,
            "redundant_size": hive.redundant_size,
            "max_epochs": ms.Master.MAX_EPOCHS,
            "min_replication_delay": MIN_REPLICATION_DELAY,
            "max_replication_delay": MAX_REPLICATION_DELAY,
            "replication_level": REPLICATION_LEVEL,
            "convergence_treshold": MIN_CONVERGENCE_THRESHOLD,
            "channel_loss": LOSS_CHANCE,
            "corruption_chance_tod": hive.corruption_chances[0]
        }

        sim_data_dict = sd.__dict__
        sim_data_dict.update(extras)
        json_string = json.dumps(
            sim_data_dict, indent=4, sort_keys=True, ensure_ascii=False)

        self.fwrite(json_string)

    def fclose(self, msg: str = None) -> None:
        """Closes the output file controlled by the FileData instance.

        Args:
             msg:
                optional; If filled, a termination message is logged into the
                output file that is being closed.
        """
        if msg:
            self.fwrite(msg)
        self.out_file.close()

    # region Overrides

    def __hash__(self):
        """Override to allows a network node object to be used as a dict key

        Returns:
            The hash of value of the referenced file :py:attr:`~name`.
        """
        return hash(str(self.name))

    def __eq__(self, other):
        """Compares if two instances of FileData are equal.

        Equality is based on name equality.

        Returns:
            True if the name attribute of both instances is the same,
            otherwise False.
        """
        if not isinstance(other, FileData):
            return False
        return self.name == other.name

    def __ne__(self, other):
        """Compares if two instances of FileData are not equal."""
        return not(self == other)

    # endregion


class FileBlockData:
    """Wrapping class for the contents of a file block.

    Among other responsabilities FileBlockData helps managing simulation
    parameters, e.g., replica control such or file block integrity.

    Attributes:
        hive_id (str):
            Unique identifier of the hive that manages the file block.
        name (str):
            The name of the file the file block belongs to.
        number (int):
            The number that uniquely identifies the file block.
        id (str):
            Concatenation the the `name` and `number`.
        references (int):
            Tracks how many references exist to the file block in the
            simulation environment. When it reaches 0 the file block ceases
            to exist and the simulation fails.
        replication_epoch (float):
            When a reference to the file block is lost, i.e., decremented,
            a replication epoch that simulates time to copy blocks from one
            node to another is assigned to this attribute.
            Until a loss occurs and after a loss is recovered,
            `recovery_epoch` is set to positive infinity.
        data (str):
            A base64-encoded string representation of the file block bytes.
        sha256 (str):
            The hash value of data resulting from a SHA256 digest.
    """

    def __init__(
            self, hive_id: str, name: str, number: int, data: bytes
    ) -> None:
        """Creates an instance of FileBlockData.

        Args:
            hive_id:
                Unique identifier of the hive that manages the file block.
            name:
                The name of the file the file block belongs to.
            number:
                The number that uniquely identifies the file block.
            data:
                Actual file block data as a sequence of bytes.
        """
        self.hive_id = hive_id
        self.name: str = name
        self.number: int = number
        self.id: str = name + "_#_" + str(number)
        self.references: int = 0
        self.replication_epoch: float = float('inf')
        self.data: str = convertions.bytes_to_base64_string(data)
        self.sha256: str = crypto.sha256(self.data)

    # region Simulation Interface
    def set_replication_epoch(self, epoch: int) -> int:
        """Sets the epoch in which replication levels should be restored.

        This method tries to assign a new epoch, in the future, at which
        recovery should be performed. If the proposition is sooner than the
        previous proposition then assignment is accepted, else, it's rejected.

        Note:
            This method of calculating the `replication_epoch` may seem
            controversial, but the justification lies in the assumption that
            if there are more network nodes monitoring file parts,
            than failure detections should be in theory, faster, unless
            complex consensus algorithms are being used between volatile
            peers, which is not our case. We assume peers only report their
            suspicions to a small number of trusted of monitors who then
            decide if the reported network node is disconnected, consequently
            losing the instance of FileBlockData and possibly others.

        Args:
            epoch:
                Simulation's current epoch.

        Returns:
            Zero if the current `replication_epoch` is positive infinity,
            otherwise the expected delay_replication is returned. This value
            can be used to log, for example, the average recovery
            delay_replication in a simulation.
        """
        new_proposed_epoch = float(
            epoch + randint(MIN_REPLICATION_DELAY, MAX_REPLICATION_DELAY))
        if new_proposed_epoch < self.replication_epoch:
            self.replication_epoch = new_proposed_epoch
        if self.replication_epoch == float('inf'):
            return 0
        else:
            return self.replication_epoch - float(epoch)

    def update_epochs_to_recover(self, epoch: int) -> None:
        """Update the `replication_epoch` after a recovery attempt was carried out.

        If the recovery attempt performed by some network node successfully
        managed to restore the replication levels to the original target, then,
        `replication_epoch` is set to positive infinity, otherwise, another
        attempt will be done in the next epoch.

        Args:
            epoch:
                Simulation's current epoch.
        """
        self.replication_epoch = float('inf') if self.references == REPLICATION_LEVEL else float(epoch + 1)

    def can_replicate(self, epoch: int) -> int:
        """Informs the calling network node if file block needs replication.

        Args:
            epoch:
                Simulation's current epoch.

        Returns:
            How many times the caller should replicate the block. The network
            node knows how many blocks he needs to create and distribute if
            returned value is bigger than zero.
        """
        if self.replication_epoch == float('inf'):
            return 0

        if 0 < self.references < REPLICATION_LEVEL and self.replication_epoch - float(epoch) <= 0.0:
            return REPLICATION_LEVEL - self.references

        return 0

    # endregion

    # region Overrides

    def __str__(self):
        """Overrides default string representation of FileBlockData instances.

        Returns:
            A dictionary representation of the object.
        """
        return "part_name: {},\npart_number: {},\npart_id: {},\npart_data: {},\nsha256: {}\n".format(self.name, self.number, self.id, self.data, self.sha256)

    # endregionss

    # region Helpers

    def decrement_and_get_references(self):
        """Decreases by one and gets the number of file block references.

        Returns:
            The number of file block references existing in the simulation
            environment.
        """
        self.references -= 1
        return self.references

    # endregion


class LoggingData:
    """Logging class that registers simulation state per epoch basis.

    Notes:
        Some attributes might not be documented, but should be straight
        forward to understand after inspecting their usage in the source code.

    Attributes:
        cswc:
            Indicates how many consecutive steps a file as been in
            convergence. Once convergence is not verified by
            :py:meth:`equal_distributions() <app.domain.cluster_groups.Cluster.equal_distributions>`
            this attribute is reseted to zero.
        largest_convergence_window:
            Stores the largest convergence window that occurred throughout
            the simulation, i.e., it stores the highest verified
            :py:attr:`~cswc`.
        convergence_set:
            Set of consecutive epochs in which convergence was verified.
            This list only stores the most up to date convergence set and like
            :py:attr:`~cswc` is cleared once convergence is not verified,
            after being appended to :py:attr:`~convergence_sets`.
        convergence_sets:
            Stores all previous convergence sets. See :py:attr:`~convergence_set`.
        terminated:
            Indicates the epoch at which the simulation was terminated.
        terminated_messages:
            Set of at least one error message that led to the failure
            of the simulation or one success message, at termination epoch
            (:py:attr:`~terminated`)
        successfull:
            When the simulation is terminated this value is set to True if
            no errors or failures occurred, i.e., if the simulation managed
            to persist the file throughout
            :py:const:`ms.Master.MAX_EPOCHS
            <environment_settings.ms.Master.MAX_EPOCHS>` time
            steps.
        blocks_corrupted:
            The number of file block repllicas lost at each epoch due
            to disk errors.
        blocks_existing:
            The number of existing file block repllicas inside the
            :py:mod:`Cluster Group <app.domain.cluster_group>` members' storage
            disks at each epoch.
        blocks_lost:
            The number of file block blocks that were lost at each epoch
            due to :py:mod:`Network Nodes <app.domain.network_nodes>` going offline.
        blocks_moved:
            The number of messages containing file block blocks that were
            transmited, including those that were not delivered or
            acknowledged, at each epoch.
        delay_replication:
            Log of the average time it took to recover one or more lost file
            block blocks during each epoch.
        delay_suspects_detection:
            Log of the time it took for each suspicious
            :py:mod:`Network Node <app.domain.network_nodes>` to be evicted
            from the :py:mod:`Cluster Group <app.domain.cluster_group>`.
        initial_spread:
            Records the strategy used distribute file blocks in the
            beggining of the simulation.
        matrices_nodes_degrees:
            Stores the in-degree and out-degree of each
            :py:mod:`Network Node <app.domain.network_nodes>` in the
            :py:mod:`Cluster Group <app.domain.cluster_group>`. One dictionary
            is kept in the list for each transition matrix used throughout
            the simulation. The integral part of the float value is the
            in-degree, the decimal part is the out-degree.
        off_node_count:
            The number of :py:mod:`Network Nodes <app.domain.network_nodes>`
            whose status changed to offline at each epoch.
        transmissions_failed:
            The number of messages transmissions that were lost in the
            network at each epoch.
    """
    # endregion

    # region Class Variables, Instance Variables and Constructors
    def __init__(self) -> None:
        """Instanciates a LoggingData object for simulation event logging."""

        max_epochs = ms.Master.MAX_EPOCHS
        max_epochs_plus_one = ms.Master.MAX_EPOCHS_PLUS_ONE

        ###############################
        # Do not alter these
        self.cswc: int = 0
        self.largest_convergence_window: int = 0
        self.convergence_set: List[int] = []
        self.convergence_sets: List[List[int]] = []
        self.terminated: int = max_epochs
        self.terminated_messages = []
        self.successfull: bool = True
        ###############################

        ###############################
        # Alter these at will
        self.blocks_corrupted: List[int] = [0] * max_epochs
        self.blocks_existing: List[int] = [0] * max_epochs
        self.blocks_lost: List[int] = [0] * max_epochs_plus_one
        self.blocks_moved: List[int] = [0] * max_epochs
        self.cluster_status_bm: List[str] = [""] * max_epochs
        self.cluster_size_bm: List[int] = [0] * max_epochs
        self.cluster_size_am: List[int] = [0] * max_epochs
        self.delay_replication: List[float] = [0.0] * max_epochs_plus_one
        self.delay_suspects_detection: Dict[int, str] = {}
        self.initial_spread = ""
        self.matrices_nodes_degrees: List[Dict[str, float]] = []
        self.off_node_count: List[int] = [0] * max_epochs
        self.transmissions_failed: List[int] = [0] * max_epochs
        ###############################

    # endregion

    # region Instance Methods

    def register_convergence(self, epoch: int) -> None:
        """Increments :py:attr:`~cswc` by one and tries to update the :py:attr:`~convergence_set`

        Checks if the counter for consecutive epoch convergence is bigger
        than the minimum threshold for verified convergence (see
        :py:const:`MIN_CONVERGENCE_THRESHOLD <environment_settings.MIN_CONVERGENCE_THRESHOLD>`
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
        rlen = self._recursive_len(self.convergence_sets)
        cw = self.largest_convergence_window
        return f"time in convergence: {rlen}\nlargest_convergence_window: {cw}"

    def __repr__(self):
        return str(self.__dict__)

    # endregion

    # region Helpers
    def log_matrices_degrees(self, nodes_degrees: Dict[str, float]):
        self.matrices_nodes_degrees.append(nodes_degrees)

    def log_replication_delay(self, delay: int, calls: int, epoch: int) -> None:
        """Logs the expected delay_replication at epoch at an epoch.

        Args:
            delay:
                The delay sum.
            calls:
                Number of times a delay_replication was generated.
            epoch:
                A simulation epoch index.
        """
        self.delay_replication[epoch - 1] = 0 if calls == 0 else delay / calls

    def log_suspicous_node_detection_delay(
            self, node_id: str, delay: int) -> None:
        """Logs the expected delay_replication at epoch at an epoch.

        Args:
            delay:
                The time it took until the specified node was evicted from a
                :py:mod:`Cluster <app.domain.cluster_groups>` after it was known
                to be offline by the perfect failure detector.
            node_id:
                A unique :py:mod:`Network Node
                <app.domain.network_nodes>` identifier.
        """
        self.delay_suspects_detection[node_id] = delay

    def log_bandwidth_units(self, n: int, epoch: int) -> None:
        """Logs the amount of moved file blocks moved at an epoch.

        Args:
            n:
                Number of parts moved at epoch.
            epoch:
                A simulation epoch index.
        """
        self.blocks_moved[epoch - 1] += n

    def log_existing_file_blocks(self, n: int, epoch: int) -> None:
        """Logs the amount of existing file blocks in the simulation environment at an epoch.

        Args:
            n:
                Number of file blocks in the system.
            epoch:
                A simulation epoch index.
        """
        self.blocks_existing[epoch - 1] += n

    def log_off_nodes(self, n: int, epoch: int) -> None:
        """Logs the amount of disconnected network_nodes at an epoch.

        Args:
            n:
                Number of disconnected network_nodes in the system.
            epoch:
                A simulation epoch index.
        """
        self.off_node_count[epoch - 1] += n

    def log_lost_file_blocks(self, n: int, epoch: int) -> None:
        """Logs the amount of permanently lost file block blocks at an epoch.

        Args:
            n:
                Number of blocks that were lost.
            epoch:
                A simulation epoch index.
        """
        self.blocks_lost[epoch - 1] += n

    def log_lost_messages(self, n: int, epoch: int) -> None:
        """Logs the amount of failed message transmissions at an epoch.

        Args:
            n:
                Number of lost terminated_messages.
            epoch:
                A simulation epoch index.
        """
        self.transmissions_failed[epoch - 1] += n

    def log_corrupted_file_blocks(self, n: int, epoch: int) -> None:
        """Logs the amount of corrupted file block blocks at an epoch.

        Args:
            n:
                Number of corrupted blocks
            epoch:
                A simulation epoch index.
        """
        self.blocks_corrupted[epoch - 1] += n

    def log_fail(self, epoch: int, message: str = "") -> None:
        """Logs the epoch at which a simulation terminated due to a failure.

        Note:
            This method should only be called when simulation terminates due
            to a failure such as a the loss of all blocks of a file block
            or the simultaneous disconnection of all network nodes in the hive.

        Args:
            message:
                optional; A log error message (default is blank)
            epoch:
                A simulation epoch at which termination occurred.
        """
        self.terminated = epoch
        self.successfull = False
        self.terminated_messages.append(message)

    def log_maintenance(self,
                        status_bm: str,
                        status_am: str,
                        size_bm: int,
                        size_am: int,
                        epoch: int) -> None:
        """Logs hive membership status and size at an epoch.

        Args:
            status_bm:
                A string that describes the status of the cluster before
                maintenance.
            status_am:
                A string that describes the status of the cluster after
                maintenance.
            size_bm:
                The number of network nodes in the hive before maintenance.
            size_am:
                The number of network nodes in the hive after maintenance.
            epoch:
                A simulation epoch at which termination occurred.
        """
        self.cluster_status_bm[epoch - 1] = status_bm
        self.cluster_size_bm[epoch - 1] = size_bm
        self.cluster_size_am[epoch - 1] = size_am
    # endregion
