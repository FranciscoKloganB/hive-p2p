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
        from other P2P member nodes within the Hive rather than having the
        BaseHive detecting the disconnection fault, i.e., Hive role in the
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

import domain.master_servers as ms
import domain.helpers.matrices as mm
from domain.network_nodes import Worker
from domain.helpers.enums import Status, HttpCodes
from domain.helpers.data_classes import FileData, FileBlockData, LoggingData
from environment_settings import REPLICATION_LEVEL, TRUE_FALSE, \
    COMMUNICATION_CHANCES, DEBUG, ABS_TOLERANCE
from utils.randoms import random_index


class Hive:
    """Represents a group of network nodes persisting a file.

    Notes:
        If you do not have a valid MatLab license you should comment
        all :py:attr:`~eng` related calls.

    Attributes:
        id:
            An uuid that uniquely identifies the Hive. Usefull for when
            there are multiple Hive instances in a simulation environment.
        current_epoch:
            The simulation's current epoch.
        corruption_chances:
            A two-element list containing the probability of file block replica
            being corrupted and not being corrupted, respectively. See
            :py:meth:`setup_epoch() <domain.cluster_groups.Hive.setup_epoch>` for
            corruption chance configuration.
        v_ (pandas DataFrame):
            Density distribution hive members must achieve with independent
            realizations for ideal persistence of the file.
        cv_ (pandas DataFrame):
            Tracks the file current density distribution, updated at each epoch.
        hivemind:
            A reference to :py:class:`~domain.master_servers.Hivemind` that
            coordinates this Hive instance.
        members:
            A collection of network nodes that belong to the Hive instance.
            See also :py:class:`~domain.domain.Worker`.
        file:
            A reference to :py:class:`~domain.helpers.FileData` object that
            represents the file being persisted by the Hive instance.
        critical_size:
            Minimum number of network nodes plus required to exist in the
            Hive to assure the target replication level.
        sufficient_size:
             Sum of :py:attr:`critical_size` and the number of nodes
             expected to fail between two successive recovery phases.
        original_size:
            The initial and optimal Hive size.
        redundant_size:
            Application-specific parameter, which indicates that membership
            of the Hive must be pruned.
        running:
            Indicates if the Hive instance is active. This attribute is
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
                 members: Dict[str, Worker],
                 sim_id: int = 0,
                 origin: str = "") -> None:
        """Instantiates an Hive object

        Args:
            hivemind:
                A reference to an :py:class:`~domain.master_servers.Hivemind`
                object that manages the Hive being initialized.
            file_name:
                The name of the file this Hive is responsible for persisting.
            members:
                A dictionary mapping unique identifiers to of the Hive's
                initial network nodes (:py:class:`~domain.domain.Worker`.)
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
        self.members: Dict[str, Worker] = members
        self.file: FileData = FileData(file_name, sim_id=sim_id, origin=origin)
        self.critical_size: int = REPLICATION_LEVEL
        self.sufficient_size: int = self.critical_size + math.ceil(len(self.members) * 0.34)
        self.original_size: int = len(members)
        self.redundant_size: int = self.sufficient_size + len(self.members)
        self.running: bool = True
        self._recovery_epoch_sum: int = 0
        self._recovery_epoch_calls: int = 0
        self.create_and_bcast_new_transition_matrix()

    # endregion

    # region Routing

    def remove_cloud_reference(self) -> None:
        """Remove cloud references and delete files within it

        Notes:
            TODO: This method requires implementation at the user descretion.
        """
        pass

    def add_cloud_reference(self) -> None:
        """Adds a cloud server reference to the membership.

        This method is used when Hive membership size becomes compromised
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
                   fresh_replica: bool = False) -> Any:
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
            return HttpCodes.DUMMY

        self.file.simulation_data.set_moved_parts_at_index(1, self.current_epoch)

        if np.random.choice(a=TRUE_FALSE, p=COMMUNICATION_CHANCES):
            self.file.simulation_data.set_lost_messages_at_index(1, self.current_epoch)
            return HttpCodes.TIME_OUT

        if not fresh_replica and np.random.choice(a=TRUE_FALSE, p=self.corruption_chances):
            self.file.simulation_data.set_corrupt_files_at_index(1, self.current_epoch)
            return HttpCodes.BAD_REQUEST

        member: Worker = self.members[destination]
        if member.status == Status.ONLINE:
            return member.receive_part(part)
        else:
            return HttpCodes.NOT_FOUND

    # endregion

    # region Swarm Guidance - Data Structure Management Only
    def new_symmetric_adjency_matrix(self, size: int):
        """Generates a random symmetric matrix

         The generated adjacency matrix does not have transient state sets or
         absorbent nodes and can effectively represent a network topology
         with bidirectional connections between network nodes.

         Args:
             size:
                The number of network nodes the Hive will have.

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
        """Sets a new desired distribution for the Hive instance.

        Normalizes the received uptimes to create a stochastic representation
        of the desired distribution, which can be used by the different
        transition matrix generation strategies.

        Args:
            member_ids:
                A list of network node identifiers currently belonging
                to the Hive membership.
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
        transition matrix the Hive is currently executing.

        Args:
            m:
                A transition matrix to be broadcasted to the network nodes
                belonging who are currently members of the Hive instance.

        Note:
            An optimization could be made that configures a transition matrix
            for the hive, independent of of file names, i.e., turn Hive
            groups into groups persisting multiple files instead of only one,
            thus reducing simulation spaceoverheads and in real-life
            scenarios, decreasing the load done to metadata servers, through
            queries and matrix calculations. For simplicity of implementation
            each Hive only manages one file for now.
        """
        for worker in self.members.values():
            transition_vector: pd.DataFrame = m.loc[:, worker.id]
            worker.set_file_routing(self.file.name, transition_vector)

    # endregion

    # region Simulation Interface

    # noinspection DuplicatedCode
    def spread_files(
            self, strategy: str, file_parts: Dict[int, FileBlockData]
    ) -> None:
        """Batch distributes files to Hive members.

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
                distributed between the Hive members according to
                the desired `strategy`.
        """
        self.file.simulation_data.initial_spread = strategy

        if strategy == "a":
            choices: List[Worker] = [*self.members.values()]
            workers: List[Worker] = np.random.choice(a=choices, size=REPLICATION_LEVEL, replace=False)
            for worker in workers:
                for part in file_parts.values():
                    part.references += 1
                    worker.receive_part(part)

        elif strategy == "u":
            for part in file_parts.values():
                choices: List[Worker] = [*self.members.values()]
                workers: List[Worker] = np.random.choice(a=choices, size=REPLICATION_LEVEL, replace=False)
                for worker in workers:
                    part.references += 1
                    worker.receive_part(part)

        elif strategy == 'i':
            choices = [*self.members.values()]
            desired_distribution: List[float] = []
            for member_id in choices:
                desired_distribution.append(self.v_.loc[member_id, 0].item())

            for part in file_parts.values():
                choices: List[Worker] = choices.copy()
                workers: List[Worker] = np.random.choice(a=choices, p=desired_distribution, size=REPLICATION_LEVEL, replace=False)
                for worker in workers:
                    part.references += 1
                    worker.receive_part(part)

    def execute_epoch(self, epoch: int) -> None:
        """Orders all network node members to execute their epoch

        Note:
            If the Hive terminates early, i.e., if it terminates before
            reaching :py:code:`~environment_settings.MAX_EPOCHS`, no logging
            should be done in :py:class:`~domain.helpers.data_classes.LoggingData`
            the received `epoch` to avoid skewing previously collected results.

        Args:
            epoch:
                The epoch the Hive should currently be in, according to it's
                managing Hivemind.

        Returns:
            False if Hive failed to persist the file it was responsible for,
            otherwise True.
        """
        self._setup_epoch(epoch)

        try:
            offline_workers: List[Worker] = self._workers_execute_epoch()
            self.evaluate_hive_convergence()
            self._membership_maintenance(offline_workers)
            if epoch == ms.Hivemind.MAX_EPOCHS:
                self.running = False
        except Exception as e:
            self.set_fail(f"Exception caused simulation termination: {str(e)}")

        self.file.simulation_data.set_delay_at_index(self._recovery_epoch_sum,
                                                     self._recovery_epoch_calls,
                                                     self.current_epoch)

    def evaluate_hive_convergence(self) -> None:
        """Verifies file block distribution and hive health status.

        This method is invoked by every Hive instance at every epoch time.
        Among other things it compares the current file block distribution
        to the desired distribution, evicts and recruits new network nodes
        for the Hive and, performs logging invocations.
        """
        if not self.members:
            self.set_fail("Hive has no remaining members.")

        parts_in_hive: int = 0
        for worker in self.members.values():
            if worker.status == Status.ONLINE:
                worker_parts_count = worker.get_file_parts_count(self.file.name)
                self.cv_.at[worker.id, 0] = worker_parts_count
                parts_in_hive += worker_parts_count
            else:
                self.cv_.at[worker.id, 0] = 0

        self.file.simulation_data.set_parts_at_index(parts_in_hive, self.current_epoch)

        if parts_in_hive <= 0:
            self.set_fail("Hive has no remaining parts.")

        self.file.parts_in_hive = parts_in_hive
        if self.equal_distributions():
            self.file.simulation_data.register_convergence(self.current_epoch)
        else:
            self.file.simulation_data.save_sets_and_reset()

    # endregion

    # region Helpers

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

    def _setup_epoch(self, epoch: int) -> None:
        """Initializes some attributes of the Hive during its initialization.

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

    def _workers_execute_epoch(self) -> List[Worker]:
        """Queries all network node members execute the epoch.

        This method logs the amount of lost parts throughout the current
        epoch according to the members who went offline and the file blocks
        they posssed and is responsible for setting up a recovery epoch those
        lost replicas (:py:meth:`domain.cluster_groups.Hive.set_recovery_epoch`).
        Similarly it logs the number of members who disconnected.

        Returns:
             A collection of members who disconnected during the current
             epoch. See :py:meth:`~domain.network_nodes.BaseNode.get_epoch_status`.
        """
        lost_parts_count: int = 0
        offline_workers: List[Worker] = []
        for worker in self.members.values():
            if worker.get_epoch_status() == Status.ONLINE:
                worker.execute_epoch(self, self.file.name)  # do not forget, file corruption, can also cause Hive failure: see Worker.discard_part(...)
            else:
                lost_parts = worker.get_file_parts(self.file.name)
                lost_parts_count += len(lost_parts)
                offline_workers.append(worker)
                for part in lost_parts.values():
                    self.set_recovery_epoch(part)
                    if part.decrement_and_get_references() == 0:
                        self.set_fail("lost all replicas of file part with id: {}".format(part.id))

        if len(offline_workers) >= len(self.members):
            self.set_fail("all hive members disconnected simultaneously")

        e: int = self.current_epoch
        sf: LoggingData = self.file.simulation_data
        sf.set_disconnected_workers_at_index(len(offline_workers), e)
        sf.set_lost_parts_at_index(lost_parts_count, e)

        return offline_workers

    def _membership_maintenance(self, offline_workers: List[Worker]) -> None:
        """Evicts disconnected workers from the Hive and attempts to recruit new ones.

        It implicitly creates a new `transition matrix` and `v_`.

        Args:
            offline_workers:
                The collection of members who disconnected during the
                current epoch.
        """
        # remove all disconnected workers from the hive
        for member in offline_workers:
            self.members.pop(member.id, None)
            member.remove_file_routing(self.file.name)

        damaged_hive_size = len(self.members)
        if damaged_hive_size >= self.sufficient_size:
            self.remove_cloud_reference()

        if damaged_hive_size >= self.redundant_size:
            status_before_recovery = "redundant"  # TODO: future-iterations evict worse members
        elif self.original_size <= damaged_hive_size < self.redundant_size:
            status_before_recovery = "stable"
        elif self.sufficient_size <= damaged_hive_size < self.original_size:
            status_before_recovery = "sufficient"
            self.members.update(self.__get_new_members__())
        elif self.critical_size < damaged_hive_size < self.sufficient_size:
            status_before_recovery = "unstable"
            self.members.update(self.__get_new_members__())
        elif 0 < damaged_hive_size <= self.critical_size:
            status_before_recovery = "critical"
            self.members.update(self.__get_new_members__())
            self.add_cloud_reference()
        else:
            status_before_recovery = "dead"

        healed_hive_size = len(self.members)
        if damaged_hive_size != healed_hive_size:
            self.create_and_bcast_new_transition_matrix()

        self.file.simulation_data.set_membership_maintenace_at_index(status_before_recovery, damaged_hive_size, healed_hive_size, self.current_epoch)

    def __get_new_members__(self) -> Dict[str, Worker]:
        """Helper method that gets adds network nodes, if possible, to the Hive.

        Returns:
            A dictionary mapping network node identifiers and their instance
            objects (:py:class:`~domain.network_nodes.BaseNode`).
        """
        return self.hivemind.find_replacement_worker(
            self.members, self.original_size - len(self.members))

    def set_fail(self, message: str) -> None:
        """Ends the Hive instance simulation.

        Sets :py:attr:`running` to False and instructs
        :py:class:`~domain.helpers.FileData.FileData` to persist
        :py:class:`~domain.helpers.data_classes.LoggingData` to disk and
        close its IO stream (py:attr:`~domain.helpers.FileData.out_file`).

        Args:
            message:
                A short explanation of why the Hive suffered early termination.
        """
        self.running = False
        self.file.simulation_data.set_fail(self.current_epoch, message)

    def set_recovery_epoch(self, part: FileBlockData) -> None:
        """Delegates to :py:meth:`~domain.helpers.FileBlockData.FileBlockData.set_recovery_epoch`

        Args:
            part: A :py:class:`~domain.helpers.FileBlockData.FileBlockData`
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
            if not np.allclose(test_target, target_distribution[0].values, atol=1e-02):
                return False
        return True

    def create_and_bcast_new_transition_matrix(self) -> None:
        """Tries to create a valid transition matrix and distributes between members of the Hive.

        After creating a transition matrix it ensures that the matrix is a
        markov matrix by invoking :py:meth:`~_validate_transition_matrix`.
        If this validation fails three times, simulation is resumed with an
        invalid matrix until the Hive membership is changed again for any
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
        self.broadcast_transition_matrix(result)  # if after 3 validations attempts no matrix was generated, use any other one.

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

    # endregion
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


