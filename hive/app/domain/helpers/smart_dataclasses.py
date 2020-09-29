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
    how many file block replicas currently existing in a 
    :py:class:`cluster group <app.domain.cluster_groups.Cluster>`
    but also keeping simulation events logged in RAM until the simulation 
    ends, at which point the logs are written to disk.

    Attributes:
        name (str):
            The name of the original file.
        existing_replicas (int):
            The number of file parts including blocks that exist for the
            named file that exist in the simulation. Updated every epoch.
        logger (:py:class:`~app.domain.helpers.smart_dataclasses.LoggingData`):
            Object that stores captured simulation data. Stored data can be
            post-processed using user defined scripts to create items such
            has graphs and figures.
        out_file (Union[str, bytes, int]):
            File output stream to where captured data is written in append
            mode and to which ``logger`` will be written to at the end of the
            simulation.
    """

    def __init__(self, name: str, sim_id: int = 0, origin: str = "") -> None:
        """Creates an instance of ``FileData``.

        Args:
            name:
                Name of the file to be referenced by the ``FileData`` object.
            sim_id:
                Identifier that generates unique output file names,
                thus guaranteeing that different simulation instances do not
                overwrite previous :py:attr:`output files <out_file>`.
            origin:
                The name of the simulation file name that started
                the simulation process. See
                :py:class:`~app.domain.master_servers.Master` and
                :py:mod:`~app.hive_simulation`. In addition to the previous,
                the origin should somehow include the cluster class name
                being run, to differentiate simulations' output files being
                executed by different distributed storage system
                implementations.
        """
        self.name: str = name
        self.existing_replicas = 0
        self.logger: LoggingData = LoggingData()
        self.out_file: IO = open(os.path.join(
            OUTFILE_ROOT, f"{Path(origin).resolve().stem}_{sim_id}.json"), "w+")

    def fwrite(self, msg: str) -> None:
        """Appends a message to the output stream of ``FileData``.

        The method automatically adds a new line character to ``msg``.

        Args:
            msg:
                The message to be logged on the :py:attr:`out_file`.
        """
        self.out_file.write(msg + "\n")

    def jwrite(self, cluster: cg.Cluster, origin: str, epoch: int) -> None:
        """Appends a json string to the output stream of ``FileData``.

        The logged data are all attributes belonging to :py:attr:`logger`.

        Args:
            cluster:
                The :py:class:`Cluster <app.domain.cluster_groups.Cluster>`
                object that manages the simulated persistence of the
                :py:attr:`named file <name>`.
            origin:
                The name of the simulation file name that started
                the simulation process. See
                :py:class:`~app.domain.master_servers.Master` and
                :py:mod:`~app.hive_simulation`.
            epoch:
                The epoch at which the :py:attr:`logger` was appended to
                :py:attr:`out_file`.
        """
        sd: LoggingData = self.logger

        sd.save_sets_and_reset()

        if not sd.terminated_messages:
            sd.terminated_messages.append("completed simulation successfully")

        sd.blocks_existing = sd.blocks_existing[:epoch]

        sd.off_node_count = sd.off_node_count[:epoch]
        sd.blocks_lost = sd.blocks_lost[:epoch]

        sd.cluster_status_bm = sd.cluster_status_bm[:epoch]
        sd.cluster_status_am = sd.cluster_status_am[:epoch]
        sd.cluster_size_bm = sd.cluster_size_bm[:epoch]
        sd.cluster_size_am = sd.cluster_size_am[:epoch]

        sd.delay_replication = sd.delay_replication[:epoch]

        sd.blocks_moved = sd.blocks_moved[:epoch]
        sd.blocks_corrupted = sd.blocks_corrupted[:epoch]
        sd.transmissions_failed = sd.transmissions_failed[:epoch]

        extras: Dict[str, Any] = {
            "cluster_type": cluster.__class__.__name__,
            "simfile_name": origin,
            "hive_id": cluster.id,
            "file_name": self.name,
            "read_size": READ_SIZE,
            "critical_size_threshold": cluster.critical_size,
            "sufficient_size_threshold": cluster.sufficient_size,
            "original_hive_size": cluster.original_size,
            "redundant_size": cluster.redundant_size,
            "max_epochs": ms.Master.MAX_EPOCHS,
            "min_replication_delay": MIN_REPLICATION_DELAY,
            "max_replication_delay": MAX_REPLICATION_DELAY,
            "replication_level": REPLICATION_LEVEL,
            "convergence_treshold": MIN_CONVERGENCE_THRESHOLD,
            "channel_loss": LOSS_CHANCE,
            "corruption_chance_tod": cluster.corruption_chances[0]
        }

        sim_data_dict = sd.__dict__
        sim_data_dict.update(extras)
        json_string = json.dumps(
            sim_data_dict, indent=4, sort_keys=True, ensure_ascii=False)

        self.fwrite(json_string)

    def fclose(self, msg: str = None) -> None:
        """Closes the output stream controlled by the ``FileData`` instance.

        Args:
             msg:
                 If filled, a termination message is appended to
                 :py:attr:`out_file`, before closing it.
        """
        if msg:
            self.fwrite(msg)
        self.out_file.close()

    # region Overrides
    def __hash__(self):
        return hash(str(self.name))

    def __eq__(self, other):
        if not isinstance(other, FileData):
            return False
        return self.name == other.name

    def __ne__(self, other):
        return not(self == other)
    # endregion


class FileBlockData:
    """Wrapping class for the contents of a file block.

    Among other responsabilities `FileBlockData` helps managing simulation
    parameters, e.g., replica control such or file block integrity.

    Attributes:
        hive_id:
            Unique identifier of the cluster that manages the file block.
        name:
            The name of the file the file block belongs to.
        number:
            The number that uniquely identifies the file block.
        id:
            Concatenation the the `name` and `number`.
        references:
            Tracks how many references exist to the file block in the
            simulation environment. When it reaches 0 the file block ceases
            to exist and the simulation fails.
        replication_epoch:
            When a reference to the file block is lost, i.e., decremented,
            a replication epoch that simulates time to copy blocks from one
            node to another is assigned to this attribute.
            Until a loss occurs and after a loss is recovered,
            `recovery_epoch` is set to positive infinity.
        data:
            A base64-encoded string representation of the file block bytes.
        sha256:
            The hash value of data resulting from a SHA256 digest.
    """

    def __init__(
            self, hive_id: str, name: str, number: int, data: bytes
    ) -> None:
        """Creates an instance of `FileBlockData`.

        Args:
            hive_id:
                Unique identifier of the cluster that manages the file block.
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
            losing the instance of `FileBlockData` and possibly others.

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
        """Overrides default string representation of `FileBlockData` instances.

        Returns:
            A dictionary representation of the object.
        """
        return (f"part_name: {self.name},\n"
                f"part_number: {self.number},\n"
                f"part_id: {self.id},\n"
                f"part_data: {self.data},\n"
                f"sha256: { self.sha256}\n")

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
    """Logger object that stores simulation events and other data.

    Note:
        Some attributes might not be documented, but should be straight
        forward to understand after inspecting their usage in the source code.

    Attributes:
        cswc (int):
            Indicates how many consecutive steps a file as been in
            convergence. Once convergence is not verified by
            :py:meth:`~app.domain.cluster_groups.Cluster.equal_distributions`
            this attribute is reset to zero.
        largest_convergence_window (int):
            Stores the largest convergence window that occurred throughout
            the simulation, i.e., it stores the highest verified
            :py:attr:`cswc`.
        convergence_set (List[int]):
            Set of consecutive epochs in which convergence was verified.
            This list only stores the most up to date convergence set and like
            :py:attr:`cswc` is cleared once convergence is not verified,
            after being appended to :py:attr:`convergence_sets`.
        convergence_sets (List[List[int]]):
            Stores all but the most recent :py:attr:`convergence_set`. If
            simulation terminates and :py:attr:`convergence_set` is not an
            empty list, that list will be appended to this one.
        terminated (int):
            Indicates the epoch at which the simulation was terminated.
        terminated_messages (List[str]):
            Set of at least one error message that led to the failure
            of the simulation or one success message, at
            :py:attr:`termination epoch <terminated>`.
        successfull (bool):
            When the simulation is :py:attr:`terminated`, this value is set
            to ``True`` if no errors or failures occurred, i.e., if the
            simulation managed to persist the file throughout the entire
            :py:const:`simulation epochs
            <app.environment_settings.ms.Master.MAX_EPOCHS>`.
        blocks_corrupted (List[int]):
            The number of :py:class:`file block replicas
            <app.domain.helpers.smart_dataclasses.FileBlockData>` lost at
            each simulation epoch due to disk errors.
        blocks_existing (List[int]):
            The number of existing :py:class:`file block replicas
            <app.domain.helpers.smart_dataclasses.FileBlockData>` inside the
            :py:mod:`cluster group <app.domain.cluster_groups>` members' storage
            disks at each epoch.
        blocks_lost (List[int]):
            The number of :py:class:`file block replicas
            <app.domain.helpers.smart_dataclasses.FileBlockData>` that were
            lost at each epoch due to :py:mod:`network nodes
            <app.domain.network_nodes>` going offline.
        blocks_moved (List[int]):
            The number of messages containing :py:class:`file block replicas
            <app.domain.helpers.smart_dataclasses.FileBlockData>` that were
            transmited, including those that were not delivered or
            acknowledged, at each epoch.
        cluster_size_bm (List[int]):
            The number of :py:mod:`network nodes <app.domain.network_nodes>`
            registered at a  :py:attr:`cluster group's members list
            <app.domain.cluster_groups.Cluster.members>`,
            before the :py:meth:`maintenance step
            <app.domain.cluster_groups.Cluster.membership_maintenance>`
            of the epoch.
        cluster_size_am (List[int]):
            The number of :py:mod:`network nodes <app.domain.network_nodes>`
            registered at a  :py:attr:`cluster group's members list
            <app.domain.cluster_groups.Cluster.members>`,
            after the :py:meth:`maintenance step
            <app.domain.cluster_groups.Cluster.membership_maintenance>`
            of the epoch.
        cluster_status_bm (List[str]):
            Strings describing the health of the :py:class:`cluster group
            <app.domain.cluster_groups.Cluster>` at each epoch,
            before the :py:meth:`maintenance step
            <app.domain.cluster_groups.Cluster.membership_maintenance>`
            of the epoch.
        cluster_status_am (List[str]):
            Strings describing the health of the :py:class:`cluster group
            <app.domain.cluster_groups.Cluster>` at each epoch,
            after the :py:meth:`maintenance step
            <app.domain.cluster_groups.Cluster.membership_maintenance>`
            of the epoch.
        delay_replication (List[float]):
            Log of the average time it took to recover one or more lost
            :py:class:`file block replicas
            <app.domain.helpers.smart_dataclasses.FileBlockData>`, at each
            epoch.
        delay_suspects_detection (Dict[int, str]):
            Log of the time it took for each suspicious
            :py:mod:`network node <app.domain.network_nodes>` to be evicted
            from the his :py:mod:`cluster group <app.domain.cluster_groups>`
            after having his :py:attr:`~app.domain.network_nodes.Node.status`
            changed from online to offline or suspicious.
        initial_spread (str):
            Records the strategy used distribute file blocks in the
            beggining of the simulation. See
            :py:meth:`~app.domain.cluster_groups.Cluster.spread_files`.
        matrices_nodes_degrees (List[Dict[str, float]]):
            Stores the ``in-degree`` and ``out-degree`` of each
            :py:mod:`network node <app.domain.network_nodes>` in the
            :py:mod:`cluster group <app.domain.cluster_groups>`. One dictionary
            is kept in the list for each transition matrix used throughout
            the simulation. The integral part of the float value is the
            in-degree, the decimal part is the out-degree.
        off_node_count (List[int]):
            The number of :py:mod:`network nodes <app.domain.network_nodes>`
            whose status changed to offline or suspicious, at each epoch.
        topologies_avg_convergence (List[float]):
            Stores floats for each of the clusters' used topologies
            representing the magnitude difference between the average density
            distribution and the desired steady state density distribution.
        transmissions_failed (List[int]):
            The number of message transmissions that were lost in the
            overlay network of a :py:mod:`cluster group
            <app.domain.cluster_groups>`, at each epoch.
    """

    # region Class Variables, Instance Variables and Constructors
    def __init__(self) -> None:
        """Instanciates a ``LoggingData`` object."""

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
        self.cluster_size_bm: List[int] = [0] * max_epochs
        self.cluster_size_am: List[int] = [0] * max_epochs
        self.cluster_status_bm: List[str] = [""] * max_epochs
        self.cluster_status_am: List[str] = [""] * max_epochs
        self.delay_replication: List[float] = [0.0] * max_epochs_plus_one
        self.delay_suspects_detection: Dict[str, int] = {}
        self.initial_spread = ""
        self.matrices_nodes_degrees: List[Dict[str, float]] = []
        self.off_node_count: List[int] = [0] * max_epochs
        self.topologies_avg_convergence: List[float] = []
        self.transmissions_failed: List[int] = [0] * max_epochs
        ###############################

    # endregion

    # region Instance Methods

    def register_convergence(self, epoch: int) -> None:
        """Increments :py:attr:`~cswc` by one and tries to update the :py:attr:`~convergence_set`

        Checks if the counter for consecutive epoch convergence is bigger
        than :py:const:`~app.environment_settings.MIN_CONVERGENCE_THRESHOLD`
        and if it is, it appends the ``epoch`` to the most recent
        :py:attr:`~convergence_set`.

        Args:
            epoch:
                The simulation epoch at which the convergence was verified.
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
            if set_len > self.largest_convergence_window:
                self.largest_convergence_window = set_len
            self.convergence_set = []
        self.cswc = 0

    def _recursive_len(self, item: Any) -> int:
        """Recusively sums the length of all lists in :py:attr:`~convergence_sets`.

        Args:
            item:
                A sub list of :py:attr:`~convergence_sets` that needs that
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
    def log_topology_avg_convergence(self, magnitude: int) -> None:
        """Logs the degree of all nodes in a Markov Matrix overlay, at the
        time of its creation, before any faults on the overlay occurs.

        Args:
            magnitude:
                The distance between the desired steady-state vector for a
                cluster's topology and the average density distribution
                vector for that same topology, assuming both were close
                to each other.
        """
        self.topologies_avg_convergence.append(magnitude)

    def log_matrices_degrees(self, nodes_degrees: Dict[str, float]):
        """Logs the degree of all nodes in a Markov Matrix overlay, at the
        time of its creation, before any faults on the overlay occurs.

        Args:
            nodes_degrees:
                A dictionary mapping the :py:attr:`node identifiers
                <app.domain.network_nodes.Node.id>` to their ``in-degree``
                and ``out-degree`` concatenated as a float.
        """
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
            or the simultaneous disconnection of all network nodes in the cluster.

        Args:
            message:
                 A log error message.
            epoch:
                A simulation epoch at which termination occurred.
        """
        self.terminated = epoch
        self.successfull = False
        self.terminated_messages.append(message)

    def log_maintenance(self,
                        size_bm: int,
                        size_am: int,
                        status_bm: str,
                        status_am: str,
                        epoch: int) -> None:
        """Logs cluster membership status and size at an epoch.

        Args:
            size_bm:
                The number of network nodes in the cluster before maintenance.
            size_am:
                The number of network nodes in the cluster after maintenance.
            status_bm:
                A string that describes the status of the cluster before
                maintenance.
            status_am:
                A string that describes the status of the cluster after
                maintenance.
            epoch:
                A simulation epoch at which termination occurred.
        """
        self.cluster_size_bm[epoch - 1] = size_bm
        self.cluster_size_am[epoch - 1] = size_am
        self.cluster_status_bm[epoch - 1] = status_bm
        self.cluster_status_am[epoch - 1] = status_am
    # endregion
