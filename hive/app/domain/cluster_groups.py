"""This module contains domain specific classes that represent groups of
storage nodes.

Classes:
    BaseHive:
        A group of P2P nodes working together to ensure the durability of a
        file using stochastic swarm guidance.
    Hive:
        A group of P2P nodes working together to ensure the durability of a
        file using stochastic swarm guidance. Differs from `BaseHive` in the
        sense that member eviction is based on the received complaints
        from other P2P member nodes within the BaseHive rather than having the
        BaseHive detecting the disconnection fault, i.e., BaseHive role in the
        simulation is more coordinative and less informative to nodes.
    Cluster:
        A group of reliable servers that ensure the durability of a file
        following a client-server model as seen in Google File System or
        Hadoop Distributed File System.
"""

from __future__ import annotations

import math
import random
import uuid
from typing import Dict, List, Any, Tuple, Optional, Union

import numpy as np
import pandas as pd
from tabulate import tabulate, JupyterHTMLStr

import domain.helpers.matrices as mm
import domain.master_servers as ms
from domain.helpers.enums import Status, HttpCodes
from domain.helpers.smart_dataclasses import FileData, FileBlockData, \
    LoggingData
from domain.network_nodes import BaseNode, HiveNode
from environment_settings import REPLICATION_LEVEL, TRUE_FALSE, \
    COMMUNICATION_CHANCES, DEBUG, ABS_TOLERANCE
from utils.randoms import random_index


class BaseHive:
    """Represents a group of network nodes persisting a file.

    Notes:
        If you do not have a valid MatLab license you should comment
        all :py:attr:`~eng` related calls.

    Attributes:
        id:
            An uuid that uniquely identifies the BaseHive. Usefull for when
            there are multiple BaseHive instances in a simulation environment.
        current_epoch:
            The simulation's current epoch.
        corruption_chances:
            A two-element list containing the probability of file block replica
            being corrupted and not being corrupted, respectively. See
            :py:meth:`setup_epoch() <domain.cluster_groups.BaseHive.setup_epoch>` for
            corruption chance configuration.
        v_ (pandas DataFrame):
            Density distribution hive members must achieve with independent
            realizations for ideal persistence of the file.
        cv_ (pandas DataFrame):
            Tracks the file current density distribution, updated at each epoch.
        hivemind:
            A reference to :py:class:`~domain.master_servers.Hivemind` that
            coordinates this BaseHive instance.
        members:
            A collection of network nodes that belong to the BaseHive instance.
            See also :py:class:`~domain.domain.BaseNode`.
        file:
            A reference to :py:class:`~domain.helpers.FileData` object that
            represents the file being persisted by the BaseHive instance.
        critical_size:
            Minimum number of network nodes plus required to exist in the
            BaseHive to assure the target replication level.
        sufficient_size:
             Sum of :py:attr:`critical_size` and the number of nodes
             expected to fail between two successive recovery phases.
        original_size:
            The initial and optimal BaseHive size.
        redundant_size:
            Application-specific parameter, which indicates that membership
            of the BaseHive must be pruned.
        running:
            Indicates if the BaseHive instance is active. This attribute is
            used by :py:class:`~domain.master_servers.Hivemind` to manage the
            simulation process.
        _recovery_epoch_sum:
            Helper attribute that facilitates the storage of the sum of the
            values returned by all :py:meth:`~FileBlockData.set_recovery_epoch`
            method calls. Important for logging purposes.
        _recovery_epoch_calls:
            Helper attribute that facilitates the storage of the sum of the
            values returned by all :py:meth:`~FileBlockData.set_recovery_epoch`
            method calls throughout the :py:attr:`~current_epoch`.
    """

    def __init__(self, hivemind: ms.Hivemind,
                 file_name: str,
                 members: Dict[str, BaseNode],
                 sim_id: int = 0,
                 origin: str = "") -> None:
        """Instantiates an `BaseHive` object

        Args:
            hivemind:
                A reference to an :py:class:`~domain.master_servers.Hivemind`
                object that manages the `BaseHive` being initialized.
            file_name:
                The name of the file this `BaseHive` is responsible
                for persisting.
            members:
                A dictionary mapping unique identifiers to of the BaseHive's
                initial network nodes (:py:class:`~domain.domain.BaseNode`.)
                to their instance objects.
            sim_id:
                optional; Identifier that generates unique output file names,
                thus guaranteeing that different simulation instances do not
                overwrite previous out files.
            origin:
                optional; The name of the simulation file name that started
                the simulation process.
        """
        self.id: str = str(uuid.uuid4())
        self.current_epoch: int = 0
        self.cv_: pd.DataFrame = pd.DataFrame()
        self.v_: pd.DataFrame = pd.DataFrame()
        self.corruption_chances: List[float] = [0, 0]
        self.hivemind = hivemind
        self.members: Dict[str, BaseNode] = members
        self.file: FileData = FileData(file_name, sim_id=sim_id, origin=origin)
        self.critical_size: int = REPLICATION_LEVEL
        self.sufficient_size: int = self.critical_size + math.ceil(len(self.members) * 0.34)
        self.original_size: int = len(members)
        self.redundant_size: int = self.sufficient_size + len(self.members)
        self.running: bool = True
        self._recovery_epoch_sum: int = 0
        self._recovery_epoch_calls: int = 0
        self.create_and_bcast_new_transition_matrix()

    # region Routing

    def remove_cloud_reference(self) -> None:
        """Remove cloud references and delete files within it

        Notes:
            TODO: This method requires implementation at the user descretion.
        """
        pass

    def add_cloud_reference(self) -> None:
        """Adds a cloud server reference to the membership.

        This method is used when BaseHive membership size becomes compromised
        and a backup solution using cloud approaches is desired. The idea
        is that surviving members upload their replicas to the cloud server,
        e.g., an Amazon S3 instance. See Hivemind method
        :py:meth:`~domain.master_servers.Hivemind.get_cloud_reference` for more
        details.

        Notes:
            TODO: This method requires implementation at the user descretion.
        """
        pass
        # noinspection PyUnusedLocal
        cloud_ref: str = self.hivemind.get_cloud_reference()

    def route_part(self,
                   sender: str,
                   destination: str,
                   part: FileBlockData,
                   fresh_replica: bool = False
                   ) -> Tuple[Union[HttpCodes, int], str]:
        """Sends one file block replica to some other network node.

        Args:
            sender:
                An identifier of the network node who is sending the message.
            destination:
                The destination network node identifier.
            part:
                The file block replica send to specified destination.
            fresh_replica:
                optional; Prevents recently created replicas from being
                corrupted, since they are not likely to be corrupted in disk.
                This argument facilitates simulation. (default: False)

        Returns:
            An HTTP code sent by destination network node.
        """
        if sender == destination:
            return HttpCodes.DUMMY, destination

        self.file.logger.log_bandwidth_units(1, self.current_epoch)

        if np.random.choice(a=TRUE_FALSE, p=COMMUNICATION_CHANCES):
            self.file.logger.log_lost_messages(1, self.current_epoch)
            return HttpCodes.TIME_OUT, destination

        if not fresh_replica and np.random.choice(a=TRUE_FALSE, p=self.corruption_chances):
            self.file.logger.log_corrupted_file_blocks(1, self.current_epoch)
            return HttpCodes.BAD_REQUEST, destination

        destination_node: BaseNode = self.members[destination]
        if destination_node.status == Status.ONLINE:
            return destination_node.receive_part(part), destination
        else:
            return HttpCodes.NOT_FOUND, destination

    # endregion

    # region Swarm Guidance - Data Structure Management Only
    def new_symmetric_adjency_matrix(self, size: int):
        """Generates a random symmetric matrix

         The generated adjacency matrix does not have transient state sets or
         absorbent nodes and can effectively represent a network topology
         with bidirectional connections between network nodes.

         Args:
             size:
                The number of network nodes the BaseHive will have.

        Returns:
            The adjency matrix representing the connections between a
            groups of network nodes.
        """
        secure_random = random.SystemRandom()
        adj_matrix: List[List[int]] = [[0] * size for _ in range(size)]
        choices: List[int] = [0, 1]

        for i in range(size):
            for j in range(i, size):
                probability = secure_random.uniform(0.0, 1.0)
                edge_val = np.random.choice(a=choices, p=[probability,
                                                          1 - probability]).item()  # converts numpy.int32 to int
                adj_matrix[i][j] = adj_matrix[j][i] = edge_val

        # Use guilty until proven innocent approach for both checks
        for i in range(size):
            is_absorbent_or_transient: bool = True
            for j in range(size):
                # Ensure state i can reach and be reached by some other state j, where i != j
                if adj_matrix[i][j] == 1 and i != j:
                    is_absorbent_or_transient = False
                    break
            if is_absorbent_or_transient:
                j = random_index(i, size)
                adj_matrix[i][j] = adj_matrix[j][i] = 1
        return adj_matrix

    def new_desired_distribution(self,
                                 member_ids: List[str],
                                 member_uptimes: List[float]) -> List[float]:
        """Sets a new desired distribution for the BaseHive instance.

        Normalizes the received uptimes to create a stochastic representation
        of the desired distribution, which can be used by the different
        transition matrix generation strategies.

        Args:
            member_ids:
                A list of network node identifiers currently belonging
                to the BaseHive membership.
            member_uptimes:
                A list in which each index contains the uptime of the network
                node with the same index in `member_ids`.

        Returns:
            A list of floats with normalized uptimes which represent the
            'reliability' of network nodes.
        """
        uptime_sum = sum(member_uptimes)
        uptimes_normalized = \
            [member_uptime / uptime_sum for member_uptime in member_uptimes]

        v_ = pd.DataFrame(data=uptimes_normalized, index=member_ids)
        self.v_ = v_
        cv_ = pd.DataFrame(data=[0] * len(v_), index=member_ids)
        self.cv_ = cv_

        return uptimes_normalized

    def new_transition_matrix(self) -> pd.DataFrame:
        """Creates a new transition matrix to be distributed among hive members.

        Returns:
            The labeled matrix that has the fastests mixing rate from all
            the pondered strategies.
        """
        member_uptimes: List[float] = []
        member_ids: List[str] = []

        for worker in self.members.values():
            member_uptimes.append(worker.uptime)
            member_ids.append(worker.id)

        A: np.ndarray = np.asarray(
            self.new_symmetric_adjency_matrix(len(member_ids)))
        v_: np.ndarray = np.asarray(
            self.new_desired_distribution(member_ids, member_uptimes))

        T = self.select_fastest_topology(A, v_)

        return pd.DataFrame(T, index=member_ids, columns=member_ids)

    def broadcast_transition_matrix(self, m: pd.DataFrame) -> None:
        """Slices a transition matrix and delivers them to respective network nodes.

        Gives each member his respective slice (vector column) of the
        transition matrix the BaseHive is currently executing.

        Args:
            m:
                A transition matrix to be broadcasted to the network nodes
                belonging who are currently members of the BaseHive instance.

        Note:
            An optimization could be made that configures a transition matrix
            for the hive, independent of of file names, i.e., turn BaseHive
            groups into groups persisting multiple files instead of only one,
            thus reducing simulation spaceoverheads and in real-life
            scenarios, decreasing the load done to metadata servers, through
            queries and matrix calculations. For simplicity of implementation
            each BaseHive only manages one file for now.
        """
        for worker in self.members.values():
            transition_vector: pd.DataFrame = m.loc[:, worker.id]
            worker.set_file_routing(self.file.name, transition_vector)

    # endregion

    # region Simulation Interface
    def setup_epoch(self, epoch: int) -> None:
        """Initializes some attributes of the BaseHive during its initialization.

        The helper method is used to isolate the initialization of some
        simulation related attributes for eaasier comprehension.

        Args:
            epoch:
                The simulation's current epoch.
        """
        self.current_epoch = epoch
        self.corruption_chances[0] = np.log10(epoch).item() / 300.0
        self.corruption_chances[1] = 1.0 - self.corruption_chances[0]
        self._recovery_epoch_sum = 0
        self._recovery_epoch_calls = 0

    def execute_epoch(self, epoch: int) -> None:
        """Orders all network node members to execute their epoch

        Note:
            If the BaseHive terminates early, i.e., if it terminates before
            reaching :py:code:`~environment_settings.MAX_EPOCHS`, no logging
            should be done in :py:class:`~domain.helpers.smart_dataclasses.LoggingData`
            the received `epoch` to avoid skewing previously collected results.

        Args:
            epoch:
                The epoch the BaseHive should currently be in, according to it's
                managing Hivemind.

        Returns:
            False if BaseHive failed to persist the file it was responsible for,
            otherwise True.
        """
        self.setup_epoch(epoch)

        try:
            off_nodes = self.nodes_execute()
            self.evaluate()
            self.maintain(off_nodes)
            if epoch == ms.Hivemind.MAX_EPOCHS:
                self.running = False
        except Exception as e:
            self.set_fail(f"Exception caused simulation termination: {str(e)}")

        self.file.logger.log_recovery_delay(self._recovery_epoch_sum,
                                            self._recovery_epoch_calls,
                                            self.current_epoch)

    def nodes_execute(self) -> List[BaseNode]:
        """Queries all network node members execute the epoch.

        This method logs the amount of lost parts throughout the current
        epoch according to the members who went offline and the file blocks
        they posssed and is responsible for setting up a recovery epoch those
        lost replicas (:py:meth:`domain.cluster_groups.BaseHive.set_recovery_epoch`).
        Similarly it logs the number of members who disconnected.

        Returns:
             A collection of members who disconnected during the current
             epoch. See :py:meth:`~domain.network_nodes.BaseNode.get_epoch_status`.
        """
        lost_parts_count: int = 0
        off_nodes: List[BaseNode] = []
        for worker in self.members.values():
            if worker.get_epoch_status() == Status.ONLINE:
                worker.execute_epoch(self, self.file.name)
            else:
                lost_parts = worker.get_file_parts(self.file.name)
                lost_parts_count += len(lost_parts)
                off_nodes.append(worker)
                for part in lost_parts.values():
                    self.set_recovery_epoch(part)
                    if part.decrement_and_get_references() == 0:
                        self.set_fail(f"Lost all replicas of file part with "
                                      f"id: {part.id}.")

        if len(off_nodes) >= len(self.members):
            self.set_fail("All hive members disconnected before maintenance.")

        sf: LoggingData = self.file.logger
        sf.log_disconnected_workers(len(off_nodes), self.current_epoch)
        sf.log_lost_file_blocks(lost_parts_count, self.current_epoch)

        return off_nodes

    def evaluate(self) -> None:
        """Verifies file block distribution and hive health status.

        This method is invoked by every BaseHive instance at every epoch time.
        Among other things it compares the current file block distribution
        to the desired distribution, evicts and recruits new network nodes
        for the BaseHive and, performs logging invocations.
        """
        if not self.members:
            self.set_fail("Cluster has no remaining members.")

        pcount: int = 0
        members = self.members.values()
        for node in members:
            if node.status == Status.ONLINE:
                worker_parts_count = node.get_file_parts_count(self.file.name)
                self.cv_.at[node.id, 0] = worker_parts_count
                pcount += worker_parts_count
            else:
                self.cv_.at[node.id, 0] = 0
        self.log_evaluation(pcount)

    def log_evaluation(self, pcount: int) -> None:
        """Helper method that performs evaluate step related logging.

        Args:
            pcount:
                The number of existing parts in the system at the
                simulation's current epoch.
        """
        self.file.logger.log_existing_file_blocks(pcount, self.current_epoch)
        if pcount <= 0:
            self.set_fail("Cluster has no remaining parts.")
        self.file.parts_in_hive = pcount
        if self.equal_distributions():
            self.file.logger.register_convergence(self.current_epoch)
        else:
            self.file.logger.save_sets_and_reset()

    def maintain(self, off_nodes: List[BaseNode]) -> None:
        """Evicts disconnected network_nodes from the BaseHive and attempts to
        recruit new ones.

        It implicitly creates a new `transition matrix` and `v_`.

        Args:
            off_nodes:
                The collection of members who disconnected during the
                current epoch.
        """
        # remove all disconnected network_nodes from the hive
        for node in off_nodes:
            self.members.pop(node.id, None)
            node.remove_file_routing(self.file.name)
        self.membership_maintenance()

    def membership_maintenance(self) -> None:
        """Attempts to recruit new :py:mod:`Network Nodes <domain.network_nodes>`"""
        damaged_hive_size = len(self.members)
        if damaged_hive_size >= self.sufficient_size:
            self.remove_cloud_reference()
        if damaged_hive_size >= self.redundant_size:
            status_bm = "redundant"
        elif self.original_size <= damaged_hive_size < self.redundant_size:
            status_bm = "stable"
        elif self.sufficient_size <= damaged_hive_size < self.original_size:
            status_bm = "sufficient"
            self.members.update(self.__get_new_members__())
        elif self.critical_size < damaged_hive_size < self.sufficient_size:
            status_bm = "unstable"
            self.members.update(self.__get_new_members__())
        elif 0 < damaged_hive_size <= self.critical_size:
            status_bm = "critical"
            self.members.update(self.__get_new_members__())
            self.add_cloud_reference()
        else:
            status_bm = "dead"

        status_am = len(self.members)
        if damaged_hive_size != status_am:
            self.create_and_bcast_new_transition_matrix()

        self.file.logger.log_maintenance(
            status_bm, damaged_hive_size, status_am, self.current_epoch)

    def complain(self, complainter: str, complainee: str) -> None:
        """Registers a complaint against a possibly offline node.

        Note:
            :py:meth:`~domain.cluster_groups.BaseHive.complain` method does
            not implement any functionality and should be overridden by any
            subclass.

        Args:
            complainter:
                The identifier of the complaining :py:mod:`Network Node
                <domain.network_nodes>`.
            complainee:
                The identifier of the :py:mod:`Network Node
                <domain.network_nodes>` being complained about.
        """
        pass
    # endregion

    # region Helpers
    def spread_files(
            self, strategy: str, file_parts: Dict[int, FileBlockData]
    ) -> None:
        """Batch distributes files to BaseHive members.

        This method is used at the start of a simulation to give all file
        blocks including the replicas to members of the hive. Different
        distribution options can be used depending on the selected `strategy`.

        Args:
            strategy:
                `u` - Distributed uniformly across network;
                `a` - Give all file block replicas to N different network
                nodes, where N is equal to :py:const:`~<environment_settings.REPLICATION_LEVEL>`;
                `i` - Distribute all file block replicas following such
                that the simulation starts with all file blocks and their
                replicas distributed with a bias towards the ideal steady
                state distribution;
            file_parts:
                A collection of file blocks, without replication, to be
                distributed between the BaseHive members according to
                the desired `strategy`.
        """
        self.file.logger.initial_spread = strategy

        if strategy == "a":
            choices: List[BaseNode] = [*self.members.values()]
            workers: List[BaseNode] = np.random.choice(a=choices, size=REPLICATION_LEVEL, replace=False)
            for worker in workers:
                for part in file_parts.values():
                    part.references += 1
                    worker.receive_part(part)

        elif strategy == "u":
            for part in file_parts.values():
                choices: List[BaseNode] = [*self.members.values()]
                workers: List[BaseNode] = np.random.choice(a=choices, size=REPLICATION_LEVEL, replace=False)
                for worker in workers:
                    part.references += 1
                    worker.receive_part(part)

        elif strategy == 'i':
            choices = [*self.members.values()]
            desired_distribution: List[float] = []
            for member_id in choices:
                desired_distribution.append(self.v_.loc[member_id, 0].item())

            for part in file_parts.values():
                choices: List[BaseNode] = choices.copy()
                workers: List[BaseNode] = np.random.choice(a=choices, p=desired_distribution, size=REPLICATION_LEVEL, replace=False)
                for worker in workers:
                    part.references += 1
                    worker.receive_part(part)

    def equal_distributions(self) -> bool:
        """Infers if v_ and cv_ are equal.

        Equalility is calculated using numpy allclose function which has the
        following formula::

            $ absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

        Returns:
            True if distributions are close enough to be considered equal,
            otherwise, it returns False.
        """
        pcount = self.file.parts_in_hive
        target = self.v_.multiply(pcount)
        rtol = self.v_[0].min()
        atol = np.clip(ABS_TOLERANCE, 0.0, 1.0) * pcount

        if DEBUG:
            print(self.__vector_comparison_table__(target, atol, rtol))

        return np.allclose(self.cv_, target, rtol=rtol, atol=atol)

    def __get_new_members__(self) -> Dict[str, BaseNode]:
        """Helper method that gets adds network nodes, if possible,
        to the BaseHive.

        Returns:
            A dictionary mapping network node identifiers and their instance
            objects (:py:class:`~domain.network_nodes.BaseNode`).
        """
        return self.hivemind.find_replacement_worker(
            self.members, self.original_size - len(self.members))

    def set_fail(self, message: str) -> None:
        """Ends the BaseHive instance simulation.

        Sets :py:attr:`running` to False and instructs
        :py:class:`~domain.helpers.smart_dataclasses.FileData` to persist
        :py:class:`~domain.helpers.smart_dataclasses.LoggingData` to disk and
        close its IO stream (py:attr:`~domain.helpers.smart_dataclasses.FileData
        .out_file`).

        Args:
            message:
                A short explanation of why the BaseHive suffered early termination.
        """
        self.running = False
        self.file.logger.log_fail(self.current_epoch, message)

    def set_recovery_epoch(self, part: FileBlockData) -> None:
        """Delegates to :py:meth:
        `~domain.helpers.smart_dataclasses.FileBlockData.set_recovery_epoch`

        Args:
            part: A :py:class:`~domain.helpers.smart_dataclasses.FileBlockData`
            instance that represents a file block replica that was lost.
        """
        self._recovery_epoch_sum += part.set_recovery_epoch(self.current_epoch)
        self._recovery_epoch_calls += 1

    def _validate_transition_matrix(self,
                                    transition_matrix: pd.DataFrame,
                                    target_distribution: pd.DataFrame) -> bool:
        """Verifies that a selected transition matrix is a Markov Matrix.

        Verification is done by raising the matrix to the power of 4096
        (just a large number) and checking if all column vectors are equal
        to the :py:attr:`~v_`.

        Returns:
            True if the matrix can converge to the desired steady state,
            otherwise False.
        """
        t_pow = np.linalg.matrix_power(transition_matrix.to_numpy(), 4096)
        column_count = t_pow.shape[1]
        for j in range(column_count):
            test_target = t_pow[:, j]  # gets array column j
            if not np.allclose(
                    test_target, target_distribution[0].values, atol=1e-02):
                return False
        return True

    def create_and_bcast_new_transition_matrix(self) -> None:
        """Tries to create a valid transition matrix and distributes between
        members of the BaseHive.

        After creating a transition matrix it ensures that the matrix is a
        markov matrix by invoking :py:meth:`~_validate_transition_matrix`.
        If this validation fails three times, simulation is resumed with an
        invalid matrix until the BaseHive membership is changed again for any
        reason.
        """
        tries = 1
        result: pd.DataFrame = pd.DataFrame()
        while tries <= 3:
            # print(f"validating transition matrix... attempt: {tries}")
            result = self.new_transition_matrix()
            if self._validate_transition_matrix(result, self.v_):
                self.broadcast_transition_matrix(result)
                break
        # Only tries to generate a valid matrix up to three times,
        # then resumes with the last generated matrix even if it never
        # converges.
        self.broadcast_transition_matrix(result)

    def select_fastest_topology(
            self, a: np.ndarray, v_: np.ndarray
    ) -> np.ndarray:
        """Creates multiple transition matrices and selects the fastest.

        The fastest of the created transition matrices corresponds to the one
        with a faster mixing rate.

        Args:
            a:
                An adjacency matrix that represents the network topology.
            v_:
                A desired distribution vector that defines the returned
                matrix steady state property.

        Returns:
            A transition matrix that is likely to be a markov matrix whose
            steady state is `v_`, but is not yet validated. See
            :py:meth:`~_validate_transition_matrix`.
        """
        results: List[Tuple[Optional[np.ndarray], float]] = [
            mm.new_mh_transition_matrix(a, v_),
            mm.new_sdp_mh_transition_matrix(a, v_),
            mm.new_go_transition_matrix(a, v_),
            mm.new_mgo_transition_matrix(a, v_)
        ]

        size = len(results)
        min_mr = float('inf')
        fastest_matrix = None
        for i in range(size):
            i_mr = results[i][1]
            if i_mr < min_mr:
                # print(f"currently selected matrix {i}")
                min_mr = i_mr
                # Worse case scenario fastest matrix will be the unoptmized MH.
                fastest_matrix = results[i][0]

        size = fastest_matrix.shape[0]
        for j in range(size):
            fastest_matrix[:, j] = np.absolute(fastest_matrix[:, j])
            fastest_matrix[:, j] /= fastest_matrix[:, j].sum()
        return fastest_matrix

    def __vector_comparison_table__(
            self, target: pd.DataFrame, atol: float, rtol: float
    ) -> Union[JupyterHTMLStr, str]:
        """Pretty prints a PSQL formatted table for visual vector comparison."""
        df = pd.DataFrame()
        df['cv_'] = self.cv_[0].values
        df['v_'] = target[0].values
        df['(cv_ - v_)'] = (self.cv_.subtract(target))[0].values
        df['tolerance'] = [(atol + np.abs(rtol) * x) for x in [*target[0]]]
        zipped = zip(df['(cv_ - v_)'].to_list(), df['tolerance'].to_list())
        df['is_close'] = [x < y for x, y in zipped]
        return tabulate(df, headers='keys', tablefmt='psql')
    # endregion


class Hive(BaseHive):
    """Represents a group of network nodes persisting a file.

    Hive instances differ from BaseHive in the sense that the
    :py:class:`~domain.domain.BaseNode` are responsible detecting that
    their cluster companions are disconnected and reporting it to the
    Hive for eviction after a certain quota is met.

    Attributes:
        complaint_threshold:
            Reference value that defines the maximum number of complaints a
            :py:mod:`Network Node <domain.network_nodes>` can receive before
            it is evicted from the Hive.
        nodes_complaints:
            A dictionary mapping :py:mod:`Network
            Nodes' <domain.network_nodes>` identifiers to their respective
            number of received complaints. When complaints becomes bigger than
            `complaint_threshold`, the respective complaintee is evicted
            from the `Hive`.
        suspicious_nodes:
            A set containing the unique identifiers of known suspicious nodes.
        _epoch_complaints:
            A set of unique identifiers formed from the concatenation of
            :py:attr:`node identifiers <domain.network_nodes.BaseNode.id>`,
            to avoid multiple complaint registrations on the same epoch,
            done by the same source towards the same target. The set is
            reset every epoch.
    """
    def __init__(self, hivemind: ms.Hivemind, file_name: str,
                 members: Dict[str, HiveNode], sim_id: int = 0,
                 origin: str = "") -> None:
        """Instantiates an `Hive` object.

        Extends:
            :py:class:`~domain.cluster_groups.BaseHive`.
        """
        super().__init__(hivemind, file_name, members, sim_id, origin)
        self.complaint_threshold: float = len(members) * 0.5
        self.nodes_complaints: Dict[str, 0] = {}
        self.suspicious_nodes: set = set()
        self._epoch_complaints: set = set()

    def execute_epoch(self, epoch: int) -> None:
        """Instructs the cluster to execute an epoch.

        Extends:
            :py:meth:`~domain.cluster_groups.BaseHive.execute_epoch`.
        """
        super().execute_epoch(epoch)
        self._epoch_complaints.clear()

    def nodes_execute(self) -> List[HiveNode]:
        """Queries all network node members execute the epoch.

        Overrides:
            :py:meth:`~domain.cluster_groups.BaseHive.nodes_execute`.
            Differs the super class implementation because it considers nodes
            as Suspects until it receives enough complaints from member nodes.
            This is important because lost parts can not be logged multiple
            times. Yet suspected network_nodes need to be contabilized as offline
            for simulation purposes without being evicted from the group
            until said detection occurs.

        Returns:
             A collection of members who disconnected during the current
             epoch. See :py:meth:`~domain.network_nodes.BaseNode.get_epoch_status`.
        """
        lost_parts_count: int = 0
        off_nodes: List[HiveNode] = []

        members = self.members.values()
        for node in members:
            node.get_epoch_status()
        for node in members:
            if node.status == Status.ONLINE:
                node.execute_epoch(self, self.file.name)
            elif node.status == Status.SUSPECT:
                lost_parts = node.get_file_parts(self.file.name)
                if node.id not in self.suspicious_nodes:
                    self.suspicious_nodes.add(node.id)
                    lost_parts_count += len(lost_parts)
                    for part in lost_parts.values():
                        if part.decrement_and_get_references() == 0:
                            self.set_fail(f"Lost all replicas of file part "
                                          f"with id: {part.id}")
                ccount = self.nodes_complaints.get(node.id, -1)
                if ccount >= self.complaint_threshold:
                    off_nodes.append(node)
                    for part in lost_parts.values():
                        self.set_recovery_epoch(part)

        if len(self.suspicious_nodes) >= len(self.members):
            self.set_fail("All hive members disconnected before maintenance.")

        sf: LoggingData = self.file.logger
        sf.log_disconnected_workers(len(off_nodes), self.current_epoch)
        sf.log_lost_file_blocks(lost_parts_count, self.current_epoch)

        return off_nodes

    def maintain(self, off_nodes: List[HiveNode]) -> None:
        """Evicts any worker whose number of complaints as surpassed the
        `complaint_threshold`.

        Overrides:
            :py:meth:`~domain.cluster_groups.BaseHive.execute_epoch`.
            Functionality is largely the same, but `off_nodes` parameter is
            ignored and replaced by an iteration over `nodes_complaints`.

        Args:
            off_nodes:
                optionalThe collection of members who disconnected during the
                current epoch.
        """
        for node in off_nodes:
            print(f"[o] Evicted suspect {node.id}.")
            self.suspicious_nodes.discard(node.id)
            self.nodes_complaints.pop(node.id, -1)
            self.members.pop(node.id, None)
            node.remove_file_routing(self.file.name)
        super().membership_maintenance()
        self.complaint_threshold = len(self.members) * 0.5

    def complain(self, complainter: str, complainee: str) -> None:
        """Registers a complaint against a possibly offline node.

        A unique identifier for the complaint is generated by concatenation
        of the complainter and the complainee unique identifiers.

        Overrides:
            :py:meth:`~domain.cluster_groups.BaseHive.complain`.
        """
        complaint_id = f"{complainter}|{complainee}"
        if complaint_id not in self._epoch_complaints:
            self._epoch_complaints.add(complaint_id)
            if complainee in self.nodes_complaints:
                self.nodes_complaints[complainee] += 1
            else:
                self.nodes_complaints[complainee] = 1
            print(f"Logged complaint {complaint_id},\n"
                  f"Complainee {complainee}, current complaints: "
                  f"{self.nodes_complaints[complainee]}")
