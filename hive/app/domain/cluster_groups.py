"""This module contains domain specific classes that represent groups of
storage nodes."""

from __future__ import annotations

import math
import uuid
from typing import Tuple, Optional, List, Dict, Any

from tabulate import tabulate

import numpy as np
import pandas as pd
import type_hints as th
import domain.master_servers as ms
import domain.helpers.enums as e
import domain.helpers.matrices as mm
import domain.helpers.smart_dataclasses as sd

from environment_settings import *

from utils.convertions import truncate_float_value


class Cluster:
    """Represents a group of network nodes ensuring the durability of a file.

    Attributes:
        id (str):
            A unique identifier of the ``Cluster`` instance.
        current_epoch (int):
            The simulation's current epoch.
        corruption_chances (List[float]):
            A two-element list containing the probability of
            :py:class:`~app.domain.helpers.smart_dataclasses.FileBlockData`
            being corrupted and not being corrupted, respectively. See
            :py:meth:`~app.domain.cluster_groups.Cluster._assign_disk_error_chance`
            for corruption chance configuration.
        master (:py:class:`~app.domain.master_servers.Master`):
            A reference to a server that coordinates or monitors the ``Cluster``.
        members (List[:py:class:`~app.domain.network_nodes.Node`]):
            A collection of network nodes that belong to the ``Cluster``.
        file (:py:class:`~app.domain.helpers.smart_dataclasses.FileData`):
            A reference to
            :py:class:`~app.domain.helpers.smart_dataclasses.FileData`
            object that represents the file being persisted by the Cluster
            instance.
        critical_size (int):
            Minimum number of network nodes plus required to exist in the
            Cluster to assure the target replication level.
        sufficient_size (int):
             Sum of :py:attr:`~app.domain.cluster_groups.Cluster.critical_size`
             and the number of nodes expected to fail between two successive
             recovery phases.
        original_size (int):
            The initial and theoretically optimal
            :py:class:`~app.domain.cluster_groups.Cluster` size.
        redundant_size (int):
            Application-specific parameter, which indicates that membership
            of the Cluster must be pruned.
        running (bool):
            Indicates if the Cluster instance is active. Used by
            :py:class:`~app.domain.master_servers.Master` to manage the
            simulation processes.
        _recovery_epoch_sum (int):
            Helper attribute that facilitates the storage of the sum of the
            values returned by all
            :py:meth:`~app.domain.helpers.smart_dataclasses.FileBlockData.set_recovery_epoch`
            method calls. Important for logging purposes.
        _recovery_epoch_calls (int):
            Helper attribute that facilitates the storage of the sum of the
            values returned by all
            :py:meth:`~app.domain.helpers.smart_dataclasses.FileBlockData.set_recovery_epoch`
            method calls throughout the :py:attr:`current_epoch`.
    """

    def __init__(self,
                 master: th.MasterType,
                 file_name: str,
                 members: th.NodeDict,
                 sim_id: int = 0,
                 origin: str = "") -> None:
        """Instantiates an ``Cluster`` object

        Args:
            master:
                A reference to an :py:class:`~app.domain.master_servers.Master`
                object that manages the ``Cluster`` being initialized.
            file_name:
                The name of the file the ``Cluster`` is responsible for
                persisting.
            members:
                A dictionary where keys are :py:attr:`node identifiers
                <app.domain.network_nodes.Node.id>` and values are their
                instance objects.
            sim_id:
                Identifier that generates unique output file names,
                thus guaranteeing that different simulation instances do not
                overwrite previous out files.
            origin:
                The name of the simulation file name that started
                the simulation process.
        """
        self.id: str = str(uuid.uuid4())
        self.current_epoch: int = 0
        self.corruption_chances: List[float] = self._assign_disk_error_chance()
        self.master = master
        self.members: th.NodeDict = members
        self.file: sd.FileData = sd.FileData(
            file_name, sim_id=sim_id, origin=origin)
        self.critical_size: int = REPLICATION_LEVEL
        expected_fails = math.ceil(len(self.members) * 0.34)
        self.sufficient_size: int = self.critical_size + expected_fails
        self.original_size: int = len(members)
        self.redundant_size: int = self.sufficient_size + len(self.members)
        self.running: bool = True
        self._recovery_epoch_sum: int = 0
        self._recovery_epoch_calls: int = 0

    # region Cluster API
    def route_part(self,
                   sender: str,
                   receiver: str,
                   replica: sd.FileBlockData,
                   fresh_replica: bool = False) -> th.HttpResponse:
        """Sends a :py:class:`file block replica
        <app.domain.helpers.smart_dataclasses.FileBlockData>` to some other
        :py:class:`network node <app.domain.network_nodes.Node>` in
        :py:attr:`members`.

        Args:
            sender:
                An identifier of the
                :py:class:`network node <app.domain.network_nodes.Node>`
                who is sending the message.
            receiver:
                The destination
                :py:class:`network node <app.domain.network_nodes.Node>`
                identifier.
            replica:
                The :py:class:`file block replica <app.domain.helpers.smart_dataclasses.FileBlockData>`
                to be sent specified destination: ``receiver``.
            fresh_replica:
                Prevents recently created replicas from being
                corrupted, since they are not likely to be corrupted in disk.
                This argument facilitates simulation.

        Returns:
            The :py:class:`http code <app.domain.helpers.enums.HttpCodes>`
            received as reply from the destination of this message.
        """
        if sender == receiver:
            return e.HttpCodes.DUMMY

        self.file.logger.log_bandwidth_units(1, self.current_epoch)

        if np.random.choice(a=TRUE_FALSE, p=COMMUNICATION_CHANCES):
            self.file.logger.log_lost_messages(1, self.current_epoch)
            return e.HttpCodes.TIME_OUT

        is_corrupted = np.random.choice(a=TRUE_FALSE, p=self.corruption_chances)
        if not fresh_replica and is_corrupted:
            self.file.logger.log_corrupted_file_blocks(1, self.current_epoch)
            return e.HttpCodes.BAD_REQUEST

        destination_node: th.NodeType = self.members[receiver]
        if destination_node.status == e.Status.ONLINE:
            return destination_node.receive_part(replica)
        else:
            return e.HttpCodes.NOT_FOUND

    def complain(
            self, complainter: str, complainee: str, reason: th.HttpResponse
    ) -> None:
        """Registers a complaint against a possibly offline node.

        Note:
            This method provides no default functionality and should be
            overridden in sub classes if required.

        Args:
            complainter:
                The identifier of the complaining :py:class:`network node
                <app.domain.network_nodes.Node>`.
            complainee:
                The identifier of the :py:class:`network node
                <app.domain.network_nodes.Node>` being complained about.
            reason:
                The :py:class:`http code <app.domain.helpers.enums.HttpCodes>`
                that led to the complaint.
        """
        pass
    # endregion

    # region Simulation setup
    def _assign_disk_error_chance(self) -> List[float]:
        """Defines the probability of a file block being corrupted while stored
        at the disk of a
        :py:class:`Network Node <app.domain.network_nodes.Node>`.

        Note:
            Recommended value should be based on the paper named
            `An Analysis of Data Corruption in the Storage Stack
            <http://www.cs.toronto.edu/bianca/papers/fast08.pdf>`. Thus
            the current implementation follows this formula:

                (:py:const:`~app.domain.master_servers.Master.MAX_EPOCHS` * ``P(Xt ≥ L)``) / :py:const:`~app.environment_settings.MONTH_EPOCHS`

            The notation ``P(Xt ≥ L)`` denotes the probability of a disk
            developing at least L checksum mismatches within T months since
            the disk’s first use in the field. Found in the paper's
            results.

        Returns:
            A two element list with respectively, the probability of losing
            and the probability of not losing a file block due to disk
            errors, at an epoch basis.
        """
        ploss_month = 0.0086
        ploss_epoch = (ms.Master.MAX_EPOCHS * ploss_month) / MONTH_EPOCHS
        ploss_epoch = truncate_float_value(ploss_epoch, 6)
        return [ploss_epoch, 1.0 - ploss_epoch]

    def _setup_epoch(self, epoch: int) -> None:
        """Initializes some attributes of the ``Cluster`` during
        its initialization.

        Args:
            epoch:
                The simulation's current epoch.
        """
        self.current_epoch = epoch
        self._recovery_epoch_sum = 0
        self._recovery_epoch_calls = 0

    def spread_files(self, replicas: th.ReplicasDict, strat: str = "i") -> None:
        """Distributes files among members of the cluster. Members are
        instances of classes belonging to module
        :py:mod:`app.domain.network_nodes`.

        Args:
            replicas:
                A collection of file replicas, without replication, to be
                distributed between the Cluster members according to
                the desired `strategy`.
            strat:
                A user defined way of identifying the way files are initially
                distributed among members of the cluster. ::

                    u
                        Distributed uniformly across network.
                    a
                        Give all file block replicas to N different network
                        nodes, where N is the replication level.
                    i
                        Distribute all file block replicas following such
                        that the simulation starts with all file replicas and
                        their replicas distributed with a bias towards the
                        ideal steady state distribution.

        """
        raise NotImplementedError("")
    # endregion

    # region Simulation steps
    def execute_epoch(self, epoch: int) -> None:
        """Orders all :py:attr:`members` to execute their epoch.

        Note:
            If the ``Cluster`` terminates early, before it reaches
            :py:const:`~app.domain.master_servers.Master.MAX_EPOCHS`,
            nothing should be logged in
            :py:class:`~app.domain.helpers.smart_dataclasses.LoggingData`
            at the specified ``epoch`` to avoid skewing previously
            collected results.

        Args:
            epoch:
                The epoch the ``Cluster`` should currently be in, according
                to it's managing :py:attr:`master` entity.

        Returns:
            False if ``Cluster`` failed to persist the :py:attr:`file` it was
            responsible for, otherwise True.
        """
        self._setup_epoch(epoch)

        off_nodes = self.nodes_execute()
        self.evaluate()
        self.maintain(off_nodes)
        if epoch == ms.Master.MAX_EPOCHS:
            self.running = False

        self.file.logger.log_replication_delay(self._recovery_epoch_sum,
                                               self._recovery_epoch_calls,
                                               self.current_epoch)

    def nodes_execute(self) -> List[th.NodeType]:
        """Queries all :py:attr:`members` to execute the epoch.

        This method logs the amount of lost replicas throughout
        :py:attr:`current_epoch` according to the :py:attr:`members` who went
        offline and the
        :py:class:`~app.domain.helpers.smart_dataclasses.FileBlockData`
        replicas they posssed and is responsible for
        :py:meth:`setting a replication epoch
        <app.domain.cluster_groups.Cluster.set_replication_epoch>`.
        Similarly it logs the number of members who disconnected.

        Returns:
             List of :py:attr:`members` that disconnected during the
             :py:attr:`current_epoch`. See
             :py:meth:`app.domain.network_nodes.Node.get_epoch_status`.

        Raises:
            NotImplementedError:
                When children of this class do not implement the abstract
                method.
        """
        raise NotImplementedError("")

    def evaluate(self) -> None:
        """Evaluates and logs the health, possibly other parameters, of the
        ``Cluster`` at every epoch.

        Raises:
            NotImplementedError:
                When children of this class do not implement the abstract
                method.
        """
        raise NotImplementedError("")

    def maintain(self, off_nodes: List[th.NodeType]) -> None:
        """Evicts disconnected :py:attr:`members` from the ``Cluster`` and
        attempts to recruit new ones.

        Args:
            off_nodes:
                The subset of :py:attr:`members` who disconnected during the
                current epoch.

        Raises:
            NotImplementedError:
                When children of this class do not implement the abstract
                method.
        """
        raise NotImplementedError("")

    def membership_maintenance(self) -> th.NodeDict:
        """Attempts to recruits new
        :py:class:`network nodes <app.domain.network_nodes.Node>` as members
        of the ``Cluster``.

        Returns:
            A dictionary that is empty if membership did not change.
        """
        sbm = len(self.members)
        status_bm = self.get_cluster_status()

        new_members: th.NodeDict = {}
        if sbm < self.original_size:
            new_members = self._get_new_members()
            self.members.update(new_members)

        sam = len(self.members)
        status_am = self.get_cluster_status()

        epoch = self.current_epoch
        self.file.logger.log_maintenance(status_bm, status_am, sbm, sam, epoch)

        return new_members
    # endregion

    # region Helpers
    def _log_evaluation(self, pcount: int) -> None:
        """Helper method that performs evaluate step related logging.

        Args:
            pcount:
                The number of existing parts in the system at the
                simulation's current epoch.
        """
        self.file.logger.log_existing_file_blocks(pcount, self.current_epoch)
        if pcount <= 0:
            self._set_fail("Cluster has no remaining parts.")
        self.file.parts_in_hive = pcount

    def _set_fail(self, message: str) -> None:
        """Ends the Cluster instance simulation.

        Sets :py:attr:`running` to False and orders
        :py:class:`~app.domain.helpers.smart_dataclasses.FileData` to write
        :py:class:`collected logs <app.domain.helpers.smart_dataclasses.LoggingData>`
        to disk and close it's
        :py:attr:`~app.domain.helpers.smart_dataclasses.FileData.out_file`
        stream.

        Args:
            message:
                A short explanation of why the ``Cluster`` terminated early.
        """
        self.running = False
        self.file.logger.log_fail(self.current_epoch, message)

    def _get_new_members(self) -> th.NodeDict:
        """Helper method that searches for possible
        :py:class:`network node <app.domain.network_nodes.Node>` by querying
        the :py:attr:`master` of the ``Cluster``.

        Returns:
            A dictionary mapping where keys are
            :py:attr:`node identifiers <app.domain.network_nodes.Node.id>`
            and values are
            :py:class:`node instances <app.domain.network_nodes.Node>`.
        """
        return self.master.find_replacement_node(
            self.members, self.original_size - len(self.members))

    def get_cluster_status(self) -> str:
        """Determines the ``Cluster``'s status based on the length of the
        current :py:attr:`members` list.

        Returns:
            The status of the ``Cluster`` as a string.
        """
        s = len(self.members)

        if s >= self.redundant_size:
            return "redundant"
        elif self.original_size <= s < self.redundant_size:
            return "stable"
        elif self.sufficient_size <= s < self.original_size:
            return "sufficient"
        elif self.critical_size < s < self.sufficient_size:
            return "unstable"
        elif 0 < s <= self.critical_size:
            return "critical"
        else:
            return "dead"

    def set_replication_epoch(self, replica: sd.FileBlockData) -> None:
        """Delegates to :py:meth:`~app.domain.helpers.smart_dataclasses.FileBlockData.set_replication_epoch`.

        Args:
            replica:
                The :py:class:`file block replica
                <app.domain.helpers.smart_dataclasses.FileBlockData>` that
                was lost.
        """
        s = replica.set_replication_epoch(self.current_epoch)
        self._recovery_epoch_sum += s
        self._recovery_epoch_calls += 1
    # endregion


class HiveCluster(Cluster):
    """Represents a group of network nodes persisting a file using swarm
    guidance algorithm.

    Attributes:
        v_ (:py:class:`~pd:pandas.DataFrame`):
            Density distribution hive members must achieve with independent
            realizations for ideal persistence of the file.
        cv_ (:py:class:`~pd:pandas.DataFrame`):
            Tracks the file current density distribution, updated at each epoch.
    """
    def __init__(self,
                 master: th.MasterType,
                 file_name: str,
                 members: th.NodeDict,
                 sim_id: int = 0,
                 origin: str = "") -> None:
        super().__init__(master, file_name, members, sim_id, origin)
        self.cv_: pd.DataFrame = pd.DataFrame()
        self.v_: pd.DataFrame = pd.DataFrame()
        self.create_and_bcast_new_transition_matrix()

    # region Simulation setup
    def spread_files(self, replicas: th.ReplicasDict, strat: str = "i") -> None:
        """Batch distributes files to Cluster members.

        This method is used at the start of a simulation to give all file
        replicas including the replicas to members of the hive. Different
        distribution options can be used depending on the selected `strategy`.
        """
        self.file.logger.initial_spread = strat

        choices: List[th.NodeType]
        selected_nodes: List[th.NodeType]
        if strat == "a":
            choices = [*self.members.values()]
            selected_nodes = np.random.choice(
                a=choices, size=REPLICATION_LEVEL, replace=False)
            for node in selected_nodes:
                for replica in replicas.values():
                    replica.references += 1
                    node.receive_part(replica)

        elif strat == "u":
            for replica in replicas.values():
                choices = [*self.members.values()]
                selected_nodes = np.random.choice(
                    a=choices, size=REPLICATION_LEVEL, replace=False)
                for node in selected_nodes:
                    replica.references += 1
                    node.receive_part(replica)

        elif strat == 'i':
            choices = [*self.members.values()]
            desired_distribution = [self.v_.loc[c.id, 0] for c in choices]
            for replica in replicas.values():
                choices_view = choices.copy()
                selected_nodes = np.random.choice(
                    a=choices_view, p=desired_distribution,
                    size=REPLICATION_LEVEL, replace=False)
                for node in selected_nodes:
                    replica.references += 1
                    node.receive_part(replica)
    # endregion

    # region Simulation steps
    def nodes_execute(self) -> List[th.NodeType]:
        """Queries all network node members execute the epoch.

        Overrides:
            :py:meth:`app.domain.cluster_groups.Cluster.nodes_execute`.

        Returns:
             A collection of members who disconnected during the current
             epoch. See
             :py:meth:`app.domain.network_nodes.Node.get_epoch_status`.
        """
        lost_parts_count: int = 0
        off_nodes: List[th.NodeType] = []
        for node in self.members.values():
            if node.get_status() == e.Status.ONLINE:
                node.execute_epoch(self, self.file.name)
            else:
                node_replicas = node.get_file_parts(self.file.name)
                lost_parts_count += len(node_replicas)
                off_nodes.append(node)
                for replica in node_replicas.values():
                    self.set_replication_epoch(replica)
                    if replica.decrement_and_get_references() == 0:
                        self._set_fail(f"Lost all replicas of file replica with "
                                       f"id: {replica.id}.")

        if len(off_nodes) >= len(self.members):
            self._set_fail("All hive members disconnected before maintenance.")

        sf: sd.LoggingData = self.file.logger
        sf.log_off_nodes(len(off_nodes), self.current_epoch)
        sf.log_lost_file_blocks(lost_parts_count, self.current_epoch)

        return off_nodes

    def evaluate(self) -> None:
        """Verifies file block distribution and hive health status.

        Among other things it compares the current file block distribution
        to the desired distribution, evicts and recruits new network nodes
        for the Cluster and, performs logging invocations.

        Overrides:
            :py:meth:`app.domain.cluster_groups.Cluster.evaluate`.
        """
        if not self.members:
            self._set_fail("Cluster has no remaining members.")

        pcount: int = 0
        members = self.members.values()
        for node in members:
            if node.status == e.Status.ONLINE:
                node_parts_count = node.get_file_parts_count(self.file.name)
                self.cv_.at[node.id, 0] = node_parts_count
                pcount += node_parts_count
            else:
                self.cv_.at[node.id, 0] = 0
        self._log_evaluation(pcount)

    def maintain(self, off_nodes: List[th.NodeType]) -> None:
        """Evicts disconnected network_nodes from the Cluster and
        attempts to recruit new ones.

        Overrides:
            :py:meth:`app.domain.cluster_groups.Cluster.maintain`.
        """
        for node in off_nodes:
            self.members.pop(node.id, None)
            node.remove_file_routing(self.file.name)
        self.membership_maintenance()

    def membership_maintenance(self) -> th.NodeDict:
        """Recruit new :py:mod:`Network Nodes <app.domain.network_nodes>`.

        Extends:
            :py:meth:`app.domain.cluster_groups.Cluster.membership_maintenance`.
            The implementation of membership_maintenance in `HiveCluster`
            class also adds and removes cloud references depending on the
            number of network nodes active in the membership before
            maintenance is performed.
        """
        s = len(self.members)
        if s <= self.critical_size:
            self.add_cloud_reference()
        elif s >= self.sufficient_size:
            self.remove_cloud_reference()

        new_members = super().membership_maintenance()
        if new_members:
            self.create_and_bcast_new_transition_matrix()
        return new_members
    # endregion

    # region Swarm guidance structure management
    def new_desired_distribution(
            self, member_ids: List[str], member_uptimes: List[float]
    ) -> List[float]:
        """Sets a new desired distribution for the Cluster instance.

        Normalizes the received uptimes to create a stochastic representation
        of the desired distribution, which can be used by the different
        transition matrix generation strategies.

        Args:
            member_ids:
                A list of network node identifiers currently belonging
                to the Cluster membership.
            member_uptimes:
                A list in which each index contains the uptime of the network
                node with the same index in `member_ids`.

        Returns:
            A list of floats with normalized uptimes which represent the
            'reliability' of network nodes.
        """
        uptime_sum = sum(member_uptimes)
        u_ = [member_uptime / uptime_sum for member_uptime in member_uptimes]

        v_ = pd.DataFrame(data=u_, index=member_ids)
        self.v_ = v_
        cv_ = pd.DataFrame(data=[0] * len(v_), index=member_ids)
        self.cv_ = cv_

        return u_

    def new_transition_matrix(self) -> pd.DataFrame:
        """Creates a new transition matrix to be distributed among hive members.

        Returns:
            The labeled matrix that has the fastests mixing rate from all
            the pondered strategies.
        """
        node_uptimes: List[float] = []
        node_ids: List[str] = []

        for node in self.members.values():
            node_uptimes.append(node.uptime)
            node_ids.append(node.id)

        size = len(node_ids)
        a = mm.new_symmetric_connected_matrix(size)
        v_ = np.asarray(self.new_desired_distribution(node_ids, node_uptimes))

        t = self.select_fastest_topology(a, v_)

        return pd.DataFrame(t, index=node_ids, columns=node_ids)

    def broadcast_transition_matrix(self, m: pd.DataFrame) -> None:
        """Slices a transition matrix and delivers them to respective
        network nodes.

        Gives each member his respective slice (vector column) of the
        transition matrix the Cluster is currently executing.

        Args:
            m:
                A transition matrix to be broadcasted to the network nodes
                belonging who are currently members of the Cluster instance.

        Note:
            An optimization could be made that configures a transition matrix
            for the hive, independent of of file names, i.e., turn Cluster
            groups into groups persisting multiple files instead of only one,
            thus reducing simulation spaceoverheads and in real-life
            scenarios, decreasing the load done to metadata servers, through
            queries and matrix calculations. For simplicity of implementation
            each Cluster only manages one file for now.
        """
        nodes_degrees: Dict[str, float] = {}
        out_degrees: pd.Series = m.apply(np.count_nonzero, axis=0)  # columns
        in_degrees: pd.Series = m.apply(np.count_nonzero, axis=1)  # rows
        for node in self.members.values():
            nid = node.id
            nodes_degrees[nid] = float(f"{in_degrees[nid]}.{out_degrees[nid]}")
            transition_vector: pd.DataFrame = m.loc[:, nid]
            node.set_file_routing(self.file.name, transition_vector)
        self.file.logger.log_matrices_degrees(nodes_degrees)

    def create_and_bcast_new_transition_matrix(self) -> None:
        """Tries to create a valid transition matrix and distributes between
        members of the Cluster.

        After creating a transition matrix it ensures that the matrix is a
        markov matrix by invoking :py:meth:`_validate_transition_matrix`.
        If this validation fails three times, simulation is resumed with an
        invalid matrix until the Cluster membership is changed again for any
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
            steady state is ``v_``, but is not yet validated. See
            :py:meth:`_validate_transition_matrix`.
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

    def _validate_transition_matrix(self,
                                    transition_matrix: pd.DataFrame,
                                    target_distribution: pd.DataFrame) -> bool:
        """Verifies that a selected transition matrix is a Markov Matrix.

        Verification is done by raising the matrix to the power of 4096
        (just a large number) and checking if all column vectors are equal
        to the :py:attr:``v_``.

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
    # endregion

    # region Cloud management
    def remove_cloud_reference(self) -> None:
        """Remove cloud references and delete files within it

        Notes:
            TODO: This method requires implementation at the user descretion.
        """
        pass

    def add_cloud_reference(self) -> None:
        """Adds a cloud server reference to the membership.

        This method is used when Cluster membership size becomes compromised
        and a backup solution using cloud approaches is desired. The idea
        is that surviving members upload their replicas to the cloud server,
        e.g., an Amazon S3 instance. See Master method
        :py:meth:`app.domain.master_servers.Master.get_cloud_reference`
        for more details.

        Notes:
            TODO: This method requires implementation at the user descretion.
        """
        pass
        # noinspection PyUnusedLocal
        cloud_ref: str = self.master.get_cloud_reference()
    # endregion

    # region Helpers
    def equal_distributions(self) -> bool:
        """Infers if :py:attr:`~app.domain.cluster_groups.HiveCluster.v_` and
        :py:attr:`~app.domain.cluster_groups.HiveCluster.cv_` are equal.

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
            print(self._pretty_print_eq_distr_table(target, atol, rtol))

        return np.allclose(self.cv_, target, rtol=rtol, atol=atol)

    def _log_evaluation(self, pcount: int) -> None:
        """Helper method that performs evaluate step related logging.

        Extends:
            :py:meth:`app.domain.cluster_groups.Cluster.__log_evaluation__`
        """
        super()._log_evaluation(pcount)
        if self.equal_distributions():
            self.file.logger.register_convergence(self.current_epoch)
        else:
            self.file.logger.save_sets_and_reset()

    def _pretty_print_eq_distr_table(
            self, target: pd.DataFrame, atol: float, rtol: float) -> Any:
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


class HiveClusterExt(HiveCluster):
    """Represents a group of network nodes persisting a file.

    HiveClusterExt instances differ from
    :py:class:`app.domain.cluster_groups.HiveCluster` because their members are
    of type :py:class:`Network Nodes <app.domain.network_nodes.HiveNodeExt>`
    . When combined these classes give nodes the responsibility of
    collaborating in the detection of faulty members of the `HiveClusterExt`
    and eventually kicking them out of the group.

    Attributes:
        complaint_threshold:
            Reference value that defines the maximum number of complaints a
            :py:mod:`Network Node <app.domain.network_nodes>` can receive before
            it is evicted from the HiveClusterExt.
        nodes_complaints:
            A dictionary mapping :py:mod:`Network
            Nodes' <app.domain.network_nodes>` identifiers to their respective
            number of received complaints. When complaints becomes bigger than
            `complaint_threshold`, the respective complaintee is evicted
            from the `HiveClusterExt`.
        suspicious_nodes:
            A dict containing the unique identifiers of known suspicious
            nodes and how many epochs have passed since they changed to that
            status.
        _epoch_complaints:
            A set of unique identifiers formed from the concatenation of
            :py:attr:`node identifiers <app.domain.network_nodes.HiveNode.id>`,
            to avoid multiple complaint registrations on the same epoch,
            done by the same source towards the same target. The set is
            reset every epoch.
    """

    def __init__(self,
                 master: th.MasterType,
                 file_name: str,
                 members: th.NodeDict,
                 sim_id: int = 0,
                 origin: str = "") -> None:
        super().__init__(master, file_name, members, sim_id, origin)
        self.complaint_threshold: float = len(members) * 0.5
        self.nodes_complaints: Dict[str, int] = {}
        self.suspicious_nodes: Dict[str, int] = {}
        self._epoch_complaints: set = set()

    # region Cluster API
    def complain(
            self, complainter: str, complainee: str, reason: th.HttpResponse
    ) -> None:
        """Registers a complaint against a possibly offline node.

        A unique identifier for the complaint is generated by concatenation
        of the complainter and the complainee unique identifiers.

        Overrides:
            :py:meth:`app.domain.cluster_groups.Cluster.complain`

        Args:
            complainter:
                The identifier of the complaining
                :py:class:`~app.domain.network_nodes.HiveNodeExt`.
            complainee:
                The identifier of the
                :py:class:`~app.domain.network_nodes.HiveNodeExt`
                being complained about.
            reason:
                The :py:class:`http code <app.domain.helpers.enums.HttpCodes>`
                that led to the complaint.
        """
        if reason == e.HttpCodes.TIME_OUT:
            return
        complaint_id = f"{complainter}|{complainee}"
        if complaint_id not in self._epoch_complaints:
            self._epoch_complaints.add(complaint_id)
            if complainee in self.nodes_complaints:
                self.nodes_complaints[complainee] += 1
            else:
                self.nodes_complaints[complainee] = 1
            print(f"    > Logged complaint {complaint_id}, "
                  f"complainee complaint count: "
                  f"{self.nodes_complaints[complainee]}")
    # endregion

    # region Simulation steps
    def execute_epoch(self, epoch: int) -> None:
        """Instructs the cluster to execute an epoch.

        Extends:
            :py:meth:`app.domain.cluster_groups.Cluster.execute_epoch`.
        """
        super().execute_epoch(epoch)
        self._epoch_complaints.clear()

    def nodes_execute(self) -> List[th.NodeType]:
        """Queries all network node members execute the epoch.

        Overrides:
            :py:meth:`app.domain.cluster_groups.HiveCluster.nodes_execute`. It
            considers nodes as Suspects until it receives enough complaints
            from member nodes. This is important because lost parts can not
            be logged multiple times. Yet suspected network_nodes need to be
            contabilized as offline for simulation purposes without being
            evicted from the group until said detection occurs.

        Returns:
             A collection of members who disconnected during the current
             epoch.
             See :py:meth:`domain.network_nodes.HiveNode.get_epoch_status`.
        """
        lost_parts_count: int = 0
        off_nodes = []

        members = self.members.values()
        for node in members:
            node.get_status()
        for node in members:
            if node.status == e.Status.ONLINE:
                node.execute_epoch(self, self.file.name)
            elif node.status == e.Status.SUSPECT:
                node_replicas = node.get_file_parts(self.file.name)
                if node.id not in self.suspicious_nodes:
                    self.suspicious_nodes[node.id] = 1
                    lost_parts_count += len(node_replicas)
                    for replica in node_replicas.values():
                        if replica.decrement_and_get_references() == 0:
                            self._set_fail(f"Lost all replicas of file replica "
                                           f"with id: {replica.id}")
                else:
                    self.suspicious_nodes[node.id] += 1

                ccount = self.nodes_complaints.get(node.id, -1)
                if ccount >= self.complaint_threshold:
                    off_nodes.append(node)
                    for replica in node_replicas.values():
                        self.set_replication_epoch(replica)

        if len(self.suspicious_nodes) >= len(self.members):
            self._set_fail("All hive members disconnected before maintenance.")

        sf: sd.LoggingData = self.file.logger
        sf.log_off_nodes(len(off_nodes), self.current_epoch)
        sf.log_lost_file_blocks(lost_parts_count, self.current_epoch)

        return off_nodes

    def maintain(self, off_nodes: List[th.NodeType]) -> None:
        """Evicts any node whose number of complaints as surpassed the
        `complaint_threshold`.

        Overrides:
            :py:meth:`domain.cluster_groups.HiveCluster.maintain`.
            Considers parameters that belong HiveClusterExt. Such as
            :py:attr:`domain.cluster_groups.HiveClusterExt.suspicious_nodes`
            and :py:attr:`domain.cluster_groups.HiveClusterExt.nodes_complaints`
        """
        for node in off_nodes:
            print(f"    [o] Evicted suspect {node.id}.")
            tte = self.suspicious_nodes.pop(node.id, -1)
            self.file.logger.log_suspicous_node_detection_delay(node.id, tte)
            self.nodes_complaints.pop(node.id, -1)
            self.members.pop(node.id, None)
            node.remove_file_routing(self.file.name)
        super().membership_maintenance()
        self.complaint_threshold = len(self.members) * 0.5
    # endregion


class HDFSCluster(Cluster):
    """Represents a group of network nodes ensuring the durability of a file
    in a Hadoop Distributed File System scenario.

    Note:
        Differs from :py:class:`~app.domain.cluster_groups.Cluster` in the sense
        that :py:class:`network nodes <app.domain.network_nodes.HDFSNode>` do not
        perform swarm guidance behaviors and instead report with regular
        heartbeats to their :py:class:`monitors
        <app.domain.cluster_groups.HDFSCluster>`. This class could be a
        *NameNode Server* in HDFS or a *master server* in GFS.

    Attributes:
        suspicious_nodes:
            A set containing the identifiers of suspicious
            :py:class:`network nodes <app.domain.network_nodes.HDFSNode>`.
        data_node_heartbeats:
            A dictionary mapping :py:attr:`node identifiers
            <app.domain.network_nodes.HDFSNode.id>` to the number of
            complaints made against them. Each node has five lives. When they
            miss five beats in a row, i.e., when the dictionary value count
            is zero, they are evicted from the cluster.
    """
    def __init__(self,
                 master: th.MasterType,
                 file_name: str,
                 members: th.NodeDict,
                 sim_id: int = 0,
                 origin: str = "") -> None:
        super().__init__(master, file_name, members, sim_id, origin)
        self.suspicious_nodes: set = set()
        self.data_node_heartbeats: Dict[str, int] = {
            node.id: 5 for node in members.values()
        }

    # region Simulation setup
    def spread_files(self, replicas: th.ReplicasDict, strat: str = "i") -> None:
        self.file.logger.initial_spread = "i"

        choices: List[th.NodeType]
        selected_nodes: List[th.NodeType]

        choices = [*self.members.values()]
        uptime_sum = sum(c.uptime for c in choices)
        chances = [c.uptime / uptime_sum for c in choices]

        for replica in replicas.values():
            choices_view = choices.copy()
            selected_nodes = np.random.choice(
                a=choices_view, p=chances,
                size=REPLICATION_LEVEL, replace=False)
            for node in selected_nodes:
                replica.references += 1
                node.receive_part(replica)
    # endregion

    # region Simulation steps
    def nodes_execute(self) -> List[th.NodeType]:
        """Queries all network node members execute the epoch.

        Overrides:
            :py:meth:`domain.cluster_groups.Cluster.nodes_execute`
            regarding the behavior of :py:mod:`Network Nodes
            <domain.network_nodes>`. They only send heartbeats to the
            `HDFSCluster` and do nothing else in their epochs unless
            specifically asked to do so.

        Returns:
             A collection of members who disconnected during the current
             epoch.
             See :py:meth:`domain.network_nodes.HiveNode.get_epoch_status`.
        """
        off_nodes = []
        lost_replicas_count: int = 0

        members = self.members.values()
        for node in members:
            node.get_status()
        for node in members:
            if node.status == e.Status.ONLINE:
                node.execute_epoch(self, self.file.name)
            elif node.status == e.Status.SUSPECT:
                # Register lost replicas the moment the node disconnects.
                if node.id not in self.suspicious_nodes:
                    self.suspicious_nodes.add(node.id)
                    node_replicas = node.get_file_parts(self.file.name)
                    lost_replicas_count += len(node_replicas)
                    for replica in node_replicas.values():
                        if replica.decrement_and_get_references() <= 0:
                            self._set_fail(f"Lost all replicas of file replica "
                                           f"with id: {replica.id}")
                # Simulate missed heartbeats.
                if node.id in self.data_node_heartbeats:
                    self.data_node_heartbeats[node.id] -= 1
                    print(f"    > Logged missed heartbeat {node.id}, "
                          f"node remaining lives: "
                          f"{self.data_node_heartbeats[node.id]}")
                    if self.data_node_heartbeats[node.id] <= 0:
                        off_nodes.append(node)
                        node_replicas = node.get_file_parts(self.file.name)
                        for replica in node_replicas.values():
                            self.set_replication_epoch(replica)

        if len(self.suspicious_nodes) >= len(self.members):
            self._set_fail("All data nodes disconnected before maintenance.")

        sf: sd.LoggingData = self.file.logger
        sf.log_off_nodes(len(off_nodes), self.current_epoch)
        sf.log_lost_file_blocks(lost_replicas_count, self.current_epoch)

        return off_nodes

    def evaluate(self) -> None:
        """Logs the number of existing replicas in the ``HDFSCluster``.

        Overrides:
            :py:meth:`app.domain.cluster_groups.Cluster.evaluate`.
        """
        if not self.members:
            self._set_fail("Cluster has no remaining members.")

        pcount: int = 0
        members = self.members.values()
        for node in members:
            if node.status == e.Status.ONLINE:
                node_replicas = node.get_file_parts_count(self.file.name)
                pcount += node_replicas
        self._log_evaluation(pcount)

    def maintain(self, off_nodes: List[th.NodeType]) -> None:
        """Evicts any :py:class:`network node <app.domain.network_nodes.HDFSNode>`
        whose heartbeats in :py:attr:`data_node_heartbeats` reached zero.

        Overrides:
            :py:meth:`app.domain.cluster_groups.Cluster.execute_epoch`.
        """
        for node in off_nodes:
            print(f"    [o] Evicted suspect {node.id}.")
            self.suspicious_nodes.discard(node.id)
            self.data_node_heartbeats.pop(node.id, -1)
            self.members.pop(node.id, None)
            self.file.logger.log_suspicous_node_detection_delay(node.id, 5)
        self.membership_maintenance()

    def membership_maintenance(self) -> th.NodeDict:
        """Recruit new :py:mod:`Network Nodes <domain.network_nodes>`.

        Extends:
            :py:meth:`domain.cluster_groups.Cluster.membership_maintenance`.
            New members are given five lives in
            :py:attr:`domain.cluster_groups.HDFSCluster.data_node_heartbeats`.
        """
        new_members = super().membership_maintenance()
        for nid in new_members.keys():
            if nid not in self.data_node_heartbeats:
                self.data_node_heartbeats[nid] = 5
    # endregion
