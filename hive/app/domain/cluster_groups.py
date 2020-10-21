"""This module contains domain specific classes that represent groups of
:py:mod:`storage nodes <app.domain.network_nodes>`."""
from __future__ import annotations

import math
import random
import uuid

from typing import Tuple, Optional, List, Dict, Any

from tabulate import tabulate

import numpy as np
import pandas as pd
import type_hints as th
import environment_settings as es
import domain.master_servers as ms
import domain.helpers.enums as e
import domain.helpers.matrices as mm
import domain.helpers.smart_dataclasses as sd


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
            :py:func:`~app.environment_settings.get_disk_error_chances`
            for corruption chance configuration.
        master (:py:class:`~app.domain.master_servers.Master`):
            A reference to a server that coordinates or monitors the ``Cluster``.
        members (:py:class:`~app.type_hints.NodeDict`):
            A collection of network nodes that belong to the ``Cluster``.
        _members_view (List[:py:class:`~app.type_hints.NodeType`]):
            A list representation of the nodes in :py:attr:`members`.
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
        _membership_changed (bool):
            Flag indicates wether or not :py:attr:`_members_view` needs
            to be updated during :py:meth:`membership_maintenance`. The
            variable is set to false at the beggining of every epoch and set
            to true if the length of ``off_nodes`` list return by
            :py:meth:`nodes_execute` is bigger than zero.
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
            master (:py:class:`~app.type_hints.MasterType`):
                A reference to an :py:class:`~app.domain.master_servers.Master`
                object that manages the ``Cluster`` being initialized.
            file_name:
                The name of the file the ``Cluster`` is responsible for
                persisting.
            members (:py:class:`~app.type_hints.NodeDict`):
                A dictionary where keys are :py:attr:`node identifiers
                <app.domain.network_nodes.Node.id>` and values are their
                :py:class:`instance objects <app.domain.network_nodes.Node>`.
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
        self.corruption_chances: List[float] = es.get_disk_error_chances(
            ms.Master.MAX_EPOCHS)

        self.master = master
        self.members: th.NodeDict = members
        self._members_view: List[th.NodeType] = list(self.members.values())

        _ = f"{self.__class__.__name__}{origin}".replace("Cluster", "-")
        self.file: sd.FileData = sd.FileData(file_name, sim_id, _)

        expected_fails = math.ceil(len(self.members) * 0.34)
        self.critical_size: int = es.REPLICATION_LEVEL
        self.sufficient_size: int = self.critical_size + expected_fails
        self.original_size: int = len(members)
        self.redundant_size: int = self.sufficient_size + len(self.members)

        self.running: bool = True
        self._membership_changed: bool = False
        self._recovery_epoch_sum: int = 0
        self._recovery_epoch_calls: int = 0

    # region Cluster API
    def route_part(self,
                   sender: str,
                   receiver: str,
                   replica: sd.FileBlockData,
                   is_fresh: bool = False) -> int:
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
            replica (:py:class:`~app.domain.helpers.smart_dataclasses.FileBlockData`):
                The :py:class:`file block replica <app.domain.helpers.smart_dataclasses.FileBlockData>`
                to be sent specified destination: ``receiver``.
            is_fresh:
                Prevents recently created replicas from being
                corrupted, since they are not likely to be corrupted in disk.
                This argument facilitates simulation.

        Returns:
            An http code sent by the ``receiver``.
        """
        if sender == receiver:
            return e.HttpCodes.DUMMY

        self.file.logger.log_bandwidth_units(1, self.current_epoch)

        tf = es.TRUE_FALSE
        if np.random.choice(a=tf, p=es.COMMUNICATION_CHANCES):
            self.file.logger.log_lost_messages(1, self.current_epoch)
            return e.HttpCodes.TIME_OUT

        is_corrupted = np.random.choice(a=tf, p=self.corruption_chances)
        if not is_fresh and is_corrupted:
            self.file.logger.log_corrupted_file_blocks(1, self.current_epoch)
            return e.HttpCodes.BAD_REQUEST

        destination_node: th.NodeType = self.members[receiver]
        if destination_node.is_up():
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
            reason (:py:data:`app.type_hints.HttpResponse`):
                The :py:class:`http code <app.domain.helpers.enums.HttpCodes>`
                that led to the complaint.
        """
        pass

    def get_node(self) -> th.NodeType:
        """Retrives a random node from the members of the cluster group,
        whose status is likely to be online.

        Returns:
            :py:class:`~app.type_hints.NodeType`:
                A random network node from :py:attr:`members`.
        """
        i = np.random.randint(0, len(self._members_view))
        candidate_node = self._members_view[i]
        return candidate_node
    # endregion

    # region Simulation setup
    def _setup_epoch(self, epoch: int) -> None:
        """Initializes some attributes cluster attributes at the start of an
        epoch.

        This method also forces all of the ``Clusters`` members to update
        their connectivity status before any node is instructed to execute.

        Args:
            epoch:
                The simulation's current epoch.
        """
        self.current_epoch = epoch
        self._membership_changed = False
        self._recovery_epoch_sum = 0
        self._recovery_epoch_calls = 0
        for member in self._members_view:
            member.update_status()

    def spread_files(self, replicas: th.ReplicasDict, strat: str = "i") -> None:
        """Distributes a collection of :py:class:`file block replicas
        <app.domain.helpers.smart_dataclasses.FileBlockData>` among the
        :py:attr:`members` of the cluster group.

        Args:
            replicas (:py:class:`~app.type_hints.ReplicasDict`):
                The :py:class:`~app.domain.helpers.smart_dataclasses.FileBlockData`
                replicas, without replication.
            strat:
                Defines how ``replicas`` will be initially distributed in
                the ``Cluster``. Unless overridden in children of this class the
                received value of ``strat`` will be ignored and will always
                be set to the default value ``i``.

                i
                    This strategy creates a probability vector
                    containing the normalization of :py:attr:`network nodes
                    uptimes' <app.domain.network_nodes.Node.uptime>` and uses
                    that vector to randomly select which
                    :py:class:`node <app.domain.network_nodes.Node>` will
                    receive each replica. There is a bias to give more
                    replicas to the most resillent :py:class:`nodes
                    <app.domain.network_nodes.Node>` which results from
                    using the created probability vector.
        """
        self.file.logger.initial_spread = "i"

        choices = self._members_view
        uptime_sum = sum(c.uptime for c in choices)
        chances = [c.uptime / uptime_sum for c in choices]

        for replica in replicas.values():
            choice_view = tuple(choices)
            selected_nodes = np.random.choice(
                a=choice_view, p=chances, size=es.REPLICATION_LEVEL, replace=False)
            for node in selected_nodes:
                replica.references += 1
                node.receive_part(replica)
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
            ``False`` if ``Cluster`` failed to persist the :py:attr:`file` it
            was responsible for, otherwise ``True``.
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
            List[:py:class:`~app.type_hints.NodeType`]:
                List of :py:attr:`members` that disconnected during the
                :py:attr:`current_epoch`. See
                :py:meth:`app.domain.network_nodes.Node.update_status`.
        """
        raise NotImplementedError("")

    def evaluate(self) -> None:
        """Evaluates and logs the health, possibly other parameters, of the
        ``Cluster`` at every epoch.
        """
        raise NotImplementedError("")

    def maintain(self, off_nodes: List[th.NodeType]) -> None:
        """Offers basic maintenance functionality for Cluster types.

        If ``off_nodes`` list param as at least one node reference,
        :py:attr:`_membership_changed` is set to ``True``.

        Args:
            off_nodes:
                A possibly empty of offline nodes.
        """
        if len(off_nodes) > 0:
            self._membership_changed = True

    def membership_maintenance(self) -> th.NodeDict:
        """Attempts to recruits new network nodes to be members of the cluster.

        The method updates both :py:attr:`members` and :py:attr:`_members_view`.

        Returns:
            :py:class:`~app.type_hints.NodeDict`:
                A dictionary that is empty if membership did not change.
        """
        sbm = len(self.members)
        status_bm = self.get_cluster_status()

        new_members: th.NodeDict = {}
        if sbm < self.original_size:
            new_members = self._get_new_members()
            if new_members:
                self.members.update(new_members)

        if self._membership_changed:
            self._members_view = list(self.members.values())  # Is this it?

        sam = len(self.members)
        status_am = self.get_cluster_status()

        epoch = self.current_epoch
        self.file.logger.log_maintenance(sbm, sam, status_bm, status_am, epoch)

        return new_members
    # endregion

    # region Helpers
    def _log_evaluation(self, plive: int, ptotal: int = -1) -> None:
        """Helper that collects ``Cluster`` data and registers it on a
        :py:class:`logger <app.domain.helpers.smart_dataclasses.LoggingData>`
        object.

        Args:
            plive:
                The number of existing parts in the cluster at the
                simulation's current epoch at online or suspect nodes.
            ptotal:
                The number of existing parts in the cluster at the
                simulation's current epoch. This parameter is optional and
                may be used or not depending on the intent of the system.
                As a rule of thumb ``plive`` tracks the number of parts that
                are alive in the system for logging purposes, where as
                ``ptotal`` is used for comparisons and averages, e.g.,
                :py:meth:`SGCluster evaluate
                <app.domain.cluster_groups.SGCluster.evaluate>`.
        """
        self.file.logger.log_existing_file_blocks(plive, self.current_epoch)
        if plive <= 0:
            self._set_fail("Cluster has no remaining parts.")
        self.file.existing_replicas = ptotal

    def _set_fail(self, message: str) -> None:
        """Ends the Cluster instance simulation.

        Sets :py:attr:`running` to ``False`` and orders
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
            :py:class:`~app.type_hints.NodeDict`:
                A dictionary mapping where keys are
                :py:attr:`node identifiers <app.domain.network_nodes.Node.id>`
                and values are
                :py:class:`node instances <app.domain.network_nodes.Node>`.
        """
        amount = self.original_size - len(self.members)
        return self.master.find_online_nodes(amount, self.members)

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


class SGCluster(Cluster):
    """Represents a group of network nodes persisting a file using swarm
    guidance algorithm.

    Attributes:
        v_ (:py:class:`~pd:pandas.DataFrame`):
            Density distribution cluster members must achieve with independent
            realizations for ideal persistence of the file.
        cv_ (:py:class:`~pd:pandas.DataFrame`):
            Tracks the file current density distribution, updated at each epoch.
        avg_ (:py:class:`~pd:pandas.DataFrame`):
            Tracks the file average density distribution. Used to assert if
            throughout the life time of a cluster, the desired density
            distribution :py:attr:`v_` was achieved on average. Differs from
            :py:attr:`cv_` because `cv_` is used for instantaneous
            convergence comparison.
        _timer (int):
            Used as a logical clock to divide the entries of :py:attr:`avg_`
            when a topology changes.
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
        self.avg_: pd.DataFrame = pd.DataFrame()
        self._timer: int = 0
        self.create_and_bcast_new_transition_matrix()

    # region Simulation setup
    def spread_files(self, replicas: th.ReplicasDict, strat: str = "i") -> None:
        """Distributes a collection of
        :py:class:`~app.domain.helpers.smart_dataclasses.FileBlockData`
        objects among the :py:attr:`~Cluster.members` of the ``SGCluster``.

        Overrides:
            :py:meth:`app.domain.cluster_groups.Cluster.spread_files`.

        Args:
            replicas (:py:class:`~app.type_hints.ReplicasDict`):
                The :py:class:`~app.domain.helpers.smart_dataclasses.FileBlockData`
                replicas, without replication.
            strat:
                Defines how ``replicas`` will be initially distributed in
                the ``Cluster``.

                u
                    Each :py:class:`file block replica
                    <app.domain.helpers.smart_dataclasses.FileBlockData>` in
                    ``replicas`` is distributed following a
                    uniform probability vector among :py:attr:`members` of
                    the cluster group.
                a
                    Each :py:class:`file block replica
                    <app.domain.helpers.smart_dataclasses.FileBlockData>`
                    in ``replicas`` is given up to ``N`` different
                    :py:attr:`members` where ``N`` is equal to
                    :py:const:`~app.environment_settings.REPLICATION_LEVEL`.
                i
                    Each :py:class:`file block replica
                    <app.domain.helpers.smart_dataclasses.FileBlockData>`
                    in ``replicas`` with bias towards the
                    ideal steady state distribution. This implementation of
                    differs from
                    :py:meth:`app.domain.cluster_groups.Cluster.spread_files`,
                    because it is not necessarely based on
                    :py:class:`node <app.domain.network_nodes.Node>` uptime.
        """
        self.file.logger.initial_spread = strat

        choices: List[th.NodeType]
        selected_nodes: List[th.NodeType]
        rl = es.REPLICATION_LEVEL
        if strat == "a":
            selected_nodes = np.random.choice(
                a=self._members_view, size=rl, replace=False)
            for node in selected_nodes:
                for replica in replicas.values():
                    replica.references += 1
                    node.receive_part(replica)

        elif strat == "u":
            for replica in replicas.values():
                selected_nodes = np.random.choice(
                    a=self._members_view, size=rl, replace=False)
                for node in selected_nodes:
                    replica.references += 1
                    node.receive_part(replica)

        elif strat == 'i':
            choices = self._members_view
            desired_distribution = [self.v_.loc[c.id, 0] for c in choices]
            for replica in replicas.values():
                choices_view = tuple(choices)
                selected_nodes = np.random.choice(
                    a=choices_view, p=desired_distribution, size=rl, replace=False)
                for node in selected_nodes:
                    replica.references += 1
                    node.receive_part(replica)
    # endregion

    # region Simulation steps
    def execute_epoch(self, epoch: int) -> None:
        self._timer += 1
        super().execute_epoch(epoch)

    def nodes_execute(self) -> List[th.NodeType]:
        """Queries all network node members execute the epoch.

        Overrides:
            :py:meth:`app.domain.cluster_groups.Cluster.nodes_execute`.

        Returns:
            List[:py:class:`~app.type_hints.NodeType`]:
                 A collection of members who disconnected during the current
                 epoch. See
                 :py:meth:`app.domain.network_nodes.Node.update_status`.
        """
        lost_parts_count: int = 0
        off_nodes: List[th.NodeType] = []

        for node in self._members_view:
            if node.is_up():
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
            self._set_fail("All cluster members disconnected before maintenance.")

        sf: sd.LoggingData = self.file.logger
        sf.log_off_nodes(len(off_nodes), self.current_epoch)
        sf.log_lost_file_blocks(lost_parts_count, self.current_epoch)

        return off_nodes

    def evaluate(self) -> None:
        if not self.members:
            self._set_fail("Cluster has no remaining members.")

        plive: int = 0
        ptotal: int = 0
        for node in self._members_view:
            c = node.get_file_parts_count(self.file.name)
            self.avg_.at[node.id, 0] += c
            self.cv_.at[node.id, 0] = c
            ptotal += c
            plive += c if node.is_up() else 0
        self._log_evaluation(plive, ptotal)

    def maintain(self, off_nodes: List[th.NodeType]) -> None:
        """Evicts any node who is referenced in off_nodes list.

        Extends:
            :py:meth:`app.domain.cluster_groups.Cluster.maintain`.

        Args:
            off_nodes (List[:py:class:`~app.type_hints.NodeType`]):
                The subset of :py:attr:`~Cluster.members` who disconnected
                during the current epoch.
        """
        if len(off_nodes) > 0:
            self._normalize_avg_()
            self._membership_changed = True
            for node in off_nodes:
                self.members.pop(node.id, None)
                node.remove_file_routing(self.file.name)
        self.membership_maintenance()

    def membership_maintenance(self) -> th.NodeDict:
        """Attempts to recruits new network nodes to be members of the cluster.

        The method updates both :py:attr:`members` and :py:attr:`_members_view`.

        Extends:
            :py:meth:`app.domain.cluster_groups.Cluster.membership_maintenance`.

            ``SGCluster.membership_maintenance`` adds and removes cloud
            references depending depending on the length of :py:attr:`~Cluster.members`
            before maintenance is performed.

        Returns:
            :py:class:`~app.type_hints.NodeDict`:
                A dictionary that is empty if membership did not change.
        """
        s = len(self.members)
        if s <= self.critical_size:
            self.add_cloud_reference()
        elif s >= self.sufficient_size:
            self.remove_cloud_reference()

        new_members = super().membership_maintenance()
        if self._membership_changed:
            self.create_and_bcast_new_transition_matrix()

        return new_members
    # endregion

    # region Swarm guidance structure management
    def new_desired_distribution(
            self, member_ids: List[str], member_uptimes: List[float]
    ) -> List[float]:
        """Sets a new :py:attr:`desired distribution <v_>` for the
        ``SGCluster``.

        Received ``member_uptimes`` are normalized to create a stochastic
        representation of the desired distribution, which can be used by the
        different transition matrix generation strategies.

        Args:
            member_ids:
                A list of :py:attr:`node identifiers
                <app.domain.network_nodes.Node.id>` who are
                :py:attr:`~Cluster.members` of the ``SGCluster``.
            member_uptimes:
                A list of :py:attr:`node identifiers
                <app.domain.network_nodes.Node.uptime>`.

        Note:
            ``member_ids`` and ``member_uptimes`` elements at each index should
            belong to each other, i.e., they should originate from from the
            same :py:class:`network node <app.domain.network_nodes.SGNode>`.
        Returns:
            A list of floats with normalized uptimes which represent the
            "reliability" of network nodes.
        """
        uptime_sum = sum(member_uptimes)
        u_ = [member_uptime / uptime_sum for member_uptime in member_uptimes]

        self.v_ = pd.DataFrame(data=u_, index=member_ids)
        self.cv_ = pd.DataFrame(data=[0] * len(self.v_), index=member_ids)
        self.avg_ = pd.DataFrame(data=[0] * len(self.v_), index=member_ids)
        self._timer = 0

        return u_

    def new_transition_matrix(self) -> pd.DataFrame:
        """Creates a new transition matrix that is likely to be a Markov Matrix.

        Returns:
            :py:class:`~pd:pandas.DataFrame`:
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
        """Slices a  matrix and delivers columns to the respective
        :py:class:`network nodes <app.domain.network_nodes.SGNode>`.

        Args:
            m (:py:class:`~pd:pandas.DataFrame`)
                A matrix to be broadcasted to the network nodes
                belonging who are currently members of the Cluster instance.

        Note:
            An optimization could be made that configures a transition matrix
            for the cluster, independent of of file names, i.e., turn cluster
            groups into groups persisting multiple files instead of only one,
            thus reducing simulation spaceoverheads and in real-life
            scenarios, decreasing the load done to metadata servers, through
            queries and matrix calculations. For simplicity of implementation
            each cluster only manages one file.
        """
        nodes_degrees: Dict[str, str] = {}
        out_degrees: pd.Series = m.apply(np.count_nonzero, axis=0)  # columns
        in_degrees: pd.Series = m.apply(np.count_nonzero, axis=1)  # rows
        for node in self.members.values():
            nid = node.id
            nodes_degrees[nid] = f"{in_degrees[nid]}i#o{out_degrees[nid]}"
            transition_vector: pd.DataFrame = m.loc[:, nid]
            node.set_file_routing(self.file.name, transition_vector)
        self.file.logger.log_matrices_degrees(nodes_degrees)

    def create_and_bcast_new_transition_matrix(self) -> None:
        """Helper method that attempts to generate a markov matrix to be
        sliced and distributed to the ``SGCluster``
        :py:attr:`~Cluster.members`.

        At most three transition matrices will be generated. The first to be
        successfully :py:meth:`validated <_validate_transition_matrix>` is
        distributed to the :py:class:`network nodes
        <app.domain.network_nodes.SGNode>`. If all matrices are invalid,
        the last matrix will be used to prevent infinite loops in the
        simulation. This is not an issue as eventually the membership of the
        ``SGCluster`` will change, thus, more opportunities to perform a
        correct swarm guidance behavior will be possible.
        """
        tries = 0
        result: pd.DataFrame = pd.DataFrame()
        while tries <= 5:
            print(f"Creating new transition matrix... try #{tries + 1}.")
            tries += 1
            result = self.new_transition_matrix()
            if self._validate_transition_matrix(result, self.v_):
                break
            print(" [x] Invalid matrix.")
        # Only tries to create a valid matrix up to five times before proceeding
        self.broadcast_transition_matrix(result)

    # noinspection PyIncorrectDocstring
    def select_fastest_topology(
            self, a: np.ndarray, v_: np.ndarray) -> np.ndarray:
        """Creates multiple transition matrices and selects the fastest.

        The fastest of the created transition matrices corresponds to the one
        with a faster mixing rate.

        Args:
            a (:py:class:`~np:numpy.ndarray`)
                An adjacency matrix that represents the network topology.
            `v_` (:py:class:`~np:numpy.ndarray`):
                A desired distribution vector that defines the returned
                matrix steady state property.

        Returns:
            :py:class:`~np:numpy.ndarray`:
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

    def _validate_transition_matrix(
            self, m: pd.DataFrame, v_: pd.DataFrame) -> bool:
        """Asserts if ``m`` is a Markov Matrix.

        Verification is done by raising the ``m`` to the power
        of ``4096`` (just a large number) and checking if all columns of the
        powered matrix are element-wise equal to the
        entries of ``target_distribution``.

        Args:
            m (:py:class:`~pd:pandas.DataFrame`):
                The matrix to be verified.
            `v_` (:py:class:`~pd:pandas.DataFrame`):
                The steady state the ``m`` is expected to have.

        Returns:
            ``True`` if the matrix converges to the ``target_distribution``,
            otherwise ``False``. I.e., if ``m`` is a
            markov matrix.
        """
        t_pow = np.linalg.matrix_power(m.to_numpy(), 4096)
        column_count = t_pow.shape[1]
        for j in range(column_count):
            test_target = t_pow[:, j]  # gets array column j
            if not np.allclose(
                    test_target, v_[0].values, atol=1e-02):
                return False
        return True
    # endregion

    # region Cloud management
    def remove_cloud_reference(self) -> None:
        """Remove cloud references and delete files within it

        Note:
            This method is virtual.
        """
        pass

    def add_cloud_reference(self) -> None:
        """Adds a cloud server to the :py:attr:`~Cluster.members` of
        the ``SGCluster``.

        This method is used when ``SGCluster`` membership size becomes
        compromised and a backup solution using cloud approaches is desired.
        The idea is that surviving members upload their replicas to the cloud
        server, e.g., an Amazon S3 instance. See Master method
        :py:meth:`~app.domain.master_servers.SGMaster.get_cloud_reference`
        for more details.

        Note:
            This method is virtual.
        """
        pass
    # endregion

    # region Helpers
    def equal_distributions(self) -> bool:
        """Asserts if the :py:attr:`desired distribution
        <app.domain.cluster_groups.SGCluster.v_>` and
        :py:attr:`current distribution
        <app.domain.cluster_groups.SGCluster.cv_>` are equal.

        Equalility is calculated using numpy allclose function which has the
        following formula: ::

            absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

        Returns:
            ``True`` if distributions are close enough to be considered equal,
            otherwise, it returns ``False``.
        """
        ptotal = self.file.existing_replicas
        target = self.v_.multiply(ptotal)
        atol = np.clip(1 / self.original_size, 0.0, es.ATOL).item() * ptotal
        return np.allclose(self.cv_, target, rtol=es.RTOL, atol=atol)

    def _log_evaluation(self, pcount: int, ptotal: int = -1) -> None:
        super()._log_evaluation(pcount, ptotal)
        if self.equal_distributions():
            self.file.logger.register_convergence(self.current_epoch)
        else:
            self.file.logger.save_sets_and_reset()

    def _normalize_avg_(self):
        self.avg_ /= self._timer
        self.avg_ /= np.sum(self.avg_)

        distance = np.abs(self.v_.subtract(self.avg_))
        magnitude = np.sqrt(distance).sum(axis=0).item()

        atol = np.clip(1 / self.original_size, 0.0, es.ATOL).item()
        goaled = np.allclose(self.avg_, self.v_, rtol=es.RTOL, atol=atol)

        self.file.logger.log_topology_goal_performance(goaled, magnitude)

    def _pretty_print_eq_distr_table(
            self, target: pd.DataFrame, rtol: float, atol: float) -> Any:
        """Pretty prints a PSQL formatted table for visual vector comparison.

        Args:
            target (:py:class:`~pd:pandas.DataFrame`):
                The :py:class:`~pd:pandas.DataFrame` object to be formatted
                as PSQL table.
            atol:
                The allowed absolute tolerance.
            rtol:
                The allowed relative tolerance.
        """
        df = pd.DataFrame()
        df['cv_'] = self.cv_[0].values
        df['v_'] = target[0].values
        df['(cv_ - v_)'] = (self.cv_.subtract(target))[0].values
        df['tolerance'] = [(atol + np.abs(rtol) * x) for x in list(target[0])]
        zipped = zip(df['(cv_ - v_)'].to_list(), df['tolerance'].to_list())
        df['is_close'] = [x < y for x, y in zipped]
        return tabulate(df, headers='keys', tablefmt='psql')
    # endregion


class SGClusterPerfect(SGCluster):
    """Represents a group of network nodes persisting a file using swarm
    guidance algorithm.

    This implementation assumes nodes never disconnect, there are no disk
    errors and there is no link loss, i.e., it is used to study properties of
    the system independently of computing environment.
    """
    def __init__(self,
                 master: th.MasterType,
                 file_name: str,
                 members: th.NodeDict,
                 sim_id: int = 0,
                 origin: str = "") -> None:
        super().__init__(master, file_name, members, sim_id, origin)
        # es.set_loss_chance(0.0)
        self.corruption_chances: List[float] = [0.0, 1.0]

    # region Swarm guidance structure management
    def new_desired_distribution(
            self, member_ids: List[str], member_uptimes: List[float]
    ) -> np.ndarray:
        """Creates a random desired distribution.

        Overrides:
            :py:meth:`app.domain.cluster_groups.SGCluster.new_desired_distribution`

        Args:
            member_ids:
                A list of :py:attr:`node identifiers
                <app.domain.network_nodes.Node.id>` who are
                :py:attr:`~Cluster.members` of the ``SGCluster``.
            member_uptimes:
                This method's parameter is ignored and can be ``None``.

        Returns:
            :py:class:`~np:numpy.ndarray`:
                A list of floats with which represent how the files should be
                distributed among network nodes in the long-run.
        """
        u_ = mm.new_vector(len(member_ids))
        self.v_ = pd.DataFrame(data=u_, index=member_ids)
        self.cv_ = pd.DataFrame(data=[0] * len(self.v_), index=member_ids)
        self.avg_ = pd.DataFrame(data=[0] * len(self.v_), index=member_ids)
        self._timer = 0

        return u_
    # endregion

    # region Simulation steps
    def execute_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        self._timer += 1

        self.nodes_execute()
        self.evaluate()

        if epoch == ms.Master.MAX_EPOCHS:
            self.running = False
            self._normalize_avg_()

    def nodes_execute(self) -> List[th.NodeType]:
        """Queries all network node members execute the epoch.

        Overrides:
            :py:meth:`app.domain.cluster_groups.Cluster.nodes_execute`.

        Returns:
            List[:py:class:`~app.type_hints.NodeType`]:
                 A collection of members who disconnected during the current
                 epoch. See
                 :py:meth:`app.domain.network_nodes.Node.update_status`.
        """
        for node in self._members_view:
            node.execute_epoch(self, self.file.name)
        return []
    # endregion

    # region Helpers
    """
    def select_fastest_topology(
            self, a: np.ndarray, v_: np.ndarray) -> np.ndarray:
        fastest_matrix, _ = mm.new_mh_transition_matrix(a, v_)
        size = fastest_matrix.shape[0]
        for j in range(size):
            fastest_matrix[:, j] = np.absolute(fastest_matrix[:, j])
            fastest_matrix[:, j] /= fastest_matrix[:, j].sum()
        return fastest_matrix
    """
    # endregion


class SGClusterExt(SGCluster):
    """Represents a group of network nodes persisting a file.

    ``SGClusterExt`` instances differ from
    :py:class:`~app.domain.cluster_groups.SGCluster` because their members are
    of type :py:class:`~app.domain.network_nodes.SGNodeExt`. When combined
    these classes give nodes the responsibility of collaborating in the
    detection of faulty members of the ``SGClusterExt`` and eventually
    kicking them out of the group.

    Attributes:
        complaint_threshold (int):
            Reference value that defines the maximum number of complaints a
            :py:class:`network node <app.domain.network_nodes.SGNodeExt>`
            can receive before it is evicted from the ``SGClusterExt``.
        nodes_complaints (Dict[str, int]):
            A dictionary mapping :py:attr:`network node identifiers'
            <app.domain.network_nodes.Node.id>` to the number of complaints
            made against them by other :py:attr:`~Cluster.members`. When
            complaints becomes bigger than py:py:attr:`complaint_threshold`
            the complaintee is evicted from the group.
        suspicious_nodes (Dict[str, int]):
            A dictionary containing the unique :py:attr:`node identifiers
            <app.domain.network_nodes.Node.id>` of known suspicious
            members and how many epochs have passed since they changed to such
            status.
        _epoch_complaints (set):
            A set of unique identifiers formed from the concatenation of
            :py:attr:`node identifiers <app.domain.network_nodes.Node.id>`,
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
        self.complaint_threshold: int = int(math.floor(len(self.members) * 0.5))
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
                :py:class:`~app.domain.network_nodes.SGNodeExt`.
            complainee:
                The identifier of the
                :py:class:`~app.domain.network_nodes.SGNodeExt`
                being complained about.
            reason (:py:data:`app.type_hints.HttpResponse`):
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
                  f"{self.nodes_complaints[complainee]} / {self.complaint_threshold}")
    # endregion

    # region Simulation steps
    def execute_epoch(self, epoch: int) -> None:
        super().execute_epoch(epoch)
        self._epoch_complaints.clear()

    def nodes_execute(self) -> List[th.NodeType]:
        """Queries all network node members execute the epoch.

        Overrides:
            :py:meth:`app.domain.cluster_groups.SGCluster.nodes_execute`.

            Offline :py:class:`network nodes <app.domain.network_nodes.SGNodeExt>`
            are considered suspects until enough complaints
            from other ``SGNodeExt`` :py:attr:`~Cluster.members` are received.
            This is important because lost parts can not be logged multiple
            times. Yet suspected :py:class:`network nodes
            <app.domain.network_nodes.SGNodeExt>` need to be contabilized
            as offline for simulation purposes without being evicted from the
            group until they are detected by their peers as being offline.

        Returns:
            List[:py:class:`~app.type_hints.NodeType`]:
                A collection of :py:attr:`~Cluster.members` who disconnected
                during the current epoch.
                See :py:meth:`app.domain.network_nodes.SGNodeExt.update_status`.
        """
        lost_parts_count: int = 0
        off_nodes = []

        for node in self._members_view:
            if node.is_up():
                node.execute_epoch(self, self.file.name)
            elif node.is_suspect() and node.id not in self.suspicious_nodes:
                self.suspicious_nodes[node.id] = self.current_epoch
                node_replicas = node.get_file_parts(self.file.name)
                lost_parts_count += len(node_replicas)
                for replica in node_replicas.values():
                    if replica.decrement_and_get_references() == 0:
                        self._set_fail(f"Lost all replicas of file replica "
                                       f"with id: {replica.id}")

        for nid, complaints in self.nodes_complaints.items():
            if complaints > self.complaint_threshold:
                node = self.members[nid]
                node_replicas = node.get_file_parts(self.file.name)
                for replica in node_replicas.values():
                    self.set_replication_epoch(replica)
                off_nodes.append(node)

        if len(self.suspicious_nodes) >= len(self.members):
            self._set_fail("All cluster members disconnected before maintenance.")

        sf: sd.LoggingData = self.file.logger
        sf.log_off_nodes(len(off_nodes), self.current_epoch)
        sf.log_lost_file_blocks(lost_parts_count, self.current_epoch)

        return off_nodes

    def maintain(self, off_nodes: List[th.NodeType]) -> None:
        """Evicts any :py:class:`network node
        <app.domain.network_nodes.SGNodeExt>` who has
        been complained about more than :py:attr:`complaint_threshold` times.

        Overrides:
            :py:meth:`app.domain.cluster_groups.Cluster.maintain`.

        Args:
            off_nodes (List[:py:class:`~app.type_hints.NodeType`]):
                The subset of :py:attr:`~Cluster.members` who disconnected
                during the current epoch.
        """
        if len(off_nodes) > 0:
            self._normalize_avg_()
            self._membership_changed = True
            for node in off_nodes:
                print(f"    [o] Evicted suspect {node.id}.")
                t = self.suspicious_nodes.pop(node.id, -1)
                self.nodes_complaints.pop(node.id, -1)
                self.members.pop(node.id, None)
                # node.remove_file_routing(self.file.name)
                if 0 < t <= self.current_epoch:
                    t = self.current_epoch - t
                    self.file.logger.log_suspicous_node_detection_delay(node.id, t)
        super().membership_maintenance()
        self.complaint_threshold = int(math.floor(len(self.members) * 0.5))
    # endregion


class HDFSCluster(Cluster):
    """Represents a group of network nodes ensuring the durability of a file
    in a Hadoop Distributed File System scenario.

    Note:
        Members of ``HDFSCluster`` are of type
        :py:class:`~app.domain.network_nodes.HDFSNode`, they do not
        perform swarm guidance behaviors and instead report with regular
        heartbeats to their :py:class:`monitors
        <app.domain.cluster_groups.HDFSCluster>`. This class could be a
        *NameNode Server* in HDFS or a *master server* in GFS.

    Attributes:
        suspicious_nodes (set):
            A set containing the identifiers of suspicious
            :py:class:`network nodes <app.domain.network_nodes.HDFSNode>`.
        data_node_heartbeats (Dict[str, int]):
            A dictionary mapping :py:attr:`node identifiers
            <app.domain.network_nodes.Node.id>` to the number of
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

    # region Simulation steps
    def nodes_execute(self) -> List[th.NodeType]:
        """Queries all :py:attr:`~Cluster.members` to execute the epoch.

        Overrides:
            :py:meth:`app.domain.cluster_groups.Cluster.nodes_execute`

        Returns:
            List[:py:class:`~app.type_hints.NodeType`]:
                A collection of :py:attr:`~Cluster.members` who disconnected
                during the current epoch. See
                :py:meth:`app.domain.network_nodes.HDFSNode.update_status`.
        """
        off_nodes = []
        lost_replicas_count: int = 0

        for node in self._members_view:
            if node.is_up():
                node.execute_epoch(self, self.file.name)
            elif node.is_suspect():
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
                self.data_node_heartbeats[node.id] -= 1
                print(f"    > Logged missed heartbeat {node.id}, node remaining"
                      f" lives: {self.data_node_heartbeats[node.id]}")
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

        plive: int = 0
        for node in self._members_view:
            if node.is_up():
                node_replicas = node.get_file_parts_count(self.file.name)
                plive += node_replicas

        self._log_evaluation(plive)

    def maintain(self, off_nodes: List[th.NodeType]) -> None:
        """Evicts any :py:class:`network node <app.domain.network_nodes.HDFSNode>`
        whose heartbeats in :py:attr:`data_node_heartbeats` reached zero.

        Extends:
            :py:meth:`app.domain.cluster_groups.Cluster.execute_epoch`.

        Args:
            off_nodes (List[:py:class:`~app.type_hints.NodeType`]):
                The subset of :py:attr:`~Cluster.members` who disconnected
                during the current epoch.
        """
        super().maintain(off_nodes)
        for node in off_nodes:
            print(f"    [o] Evicted suspect {node.id}.")
            self.suspicious_nodes.discard(node.id)
            self.data_node_heartbeats.pop(node.id, -1)
            self.members.pop(node.id, None)
            self.file.logger.log_suspicous_node_detection_delay(node.id, 5)
        self.membership_maintenance()

    def membership_maintenance(self) -> th.NodeDict:
        new_members = super().membership_maintenance()
        for nid in new_members:
            self.data_node_heartbeats[nid] = 5
    # endregion


class NewscastCluster(Cluster):
    """Represents a P2P network of nodes performing mean degree aggregation,
    while simultaneously using Newscast for ``view shuffling``.
    """

    def __init__(self,
                 master: th.MasterType,
                 file_name: str,
                 members: th.NodeDict,
                 sim_id: int = 0,
                 origin: str = "") -> None:
        super().__init__(master, file_name, members, sim_id, origin)

    # region Cluster API
    # noinspection PyAttributeOutsideInit
    def log_aggregation(self, value: float):
        if value < self.min:
            self.min = value
            self.count_min = 1
        elif value == self.min:
            self.count_min += 1

        if value > self.max:
            self.max = value
            self.count_max = 1
        elif value == self.max:
            self.count_max += 1

        self.n += 1
        self.sum += value
        self.sqrsum += value * value

    def wire_k_out(self):
        """Creates a random directed P2P topology.

        The initial cache size of each :py:class:`network node
        <app.domain.network_nodes.NewscastNode>`, is at most as big as
        :py:const:`~app.environment_settings.NEWSCAST_CACHE_SIZE`.

        Note:
            The topology does not have self loops, because
            :py:meth:`~app.domain.network_nodes.NewscastNode.add_neighbor`
            does not accept node self addition to
            :py:attr:`~app.domain.network_nodes.NewscastNode.view`. In rare
            occasions, the selected node out-going edges might all be
            invalid, this should be a non-issue, as the nodes will eventually
            join the overaly throughout the simulation.
        """
        network_size = len(self._members_view)
        for i in range(network_size):
            s = np.random.randint(0, network_size, size=es.NEWSCAST_CACHE_SIZE)
            s = list(dict.fromkeys(s))
            member = self._members_view[i]
            for j in s:
                another_member = self._members_view[j]
                member.add_neighbor(another_member)

    # endregion

    # region Simulation steps
    def execute_epoch(self, epoch: int) -> None:
        self._setup_epoch(epoch)
        self.nodes_execute()
        self.evaluate()
        if epoch == ms.Master.MAX_EPOCHS:
            self.running = False

    def nodes_execute(self) -> Optional[List[th.NodeType]]:
        """Queries all network node members execute the epoch.

        Overrides:
            :py:meth:`app.domain.cluster_groups.Cluster.nodes_execute`.

            Note:
                :py:meth:`NewscasterCluster.nodes_execute
                <app.domain.cluster_groups.NewscastNode.nodes_execute>`
                always returns None.

        Returns:
            List[:py:class:`~app.type_hints.NodeType`]:
                 A collection of members who disconnected during the current
                 epoch. See
                 :py:meth:`app.domain.network_nodes.NewscastNode.update_status`.
        """
        random.shuffle(self._members_view)

        for node in self._members_view:
            node.execute_epoch(self, self.file.name)
        return None

    def evaluate(self) -> None:
        """Prints the epoch's aggregated peer degree, to the command-line
        interface.
        """
        print({
            "min": self.min,
            "max": self.max,
            "sum": self.sum / self.n,
            "n": self.n,
            "count_min": self.count_min,
            "count_max": self.count_max
        })
    # endregion

    # region Simulation setup
    def _setup_epoch(self, epoch: int) -> None:
        """Initializes some attributes cluster attributes at the start of an
        epoch.

        Extends:
            :py:meth:`app.domain.cluster_groups.Cluster._setup_epoch`

        Args:
            epoch:
                The simulation's current epoch.
        """
        super()._setup_epoch(epoch)
        self.min: float = float('inf')
        self.max: float = 0.0
        self.sum: float = 0.0
        self.sqrsum: float = 0.0
        self.n: int = 0
        self.count_min: float = 0.0
        self.count_max: float = 0.0

    def spread_files(self, replicas: th.ReplicasDict, strat: str = "o") -> None:
        """Distributes a collection of :py:class:`file block replicas
        <app.domain.helpers.smart_dataclasses.FileBlockData>` among the
        :py:attr:`members` of the cluster group.

        Overrides:
            :py:meth:`app.dommain.cluster_groups.Cluster.spread_files`

        Args:
            replicas (:py:class:`~app.type_hints.ReplicasDict`):
                The :py:class:`~app.domain.helpers.smart_dataclasses.FileBlockData`
                replicas, without replication.
            strat:
                Defines how ``replicas`` will be initially distributed in
                the ``Cluster``. Unless overridden in children of this class the
                received value of ``strat`` will be ignored and will always
                be set to the default value ``o``.

                o
                    This strategy assumes erasure-coding is being used and
                    that each :py:class:`network node
                    <app.domain.network_nodes.Node>` will have no more than
                    one encoded block, i.e., replication level is always
                    equal to one. Note however, that if there are more encoded
                    blocks than there are :py:class:`network nodes
                    <app.domain.network_nodes.Node>`, some of these ``nodes``
                    might end up possessing an excessive amount of blocks.
        """
        self.file.logger.initial_spread = "o"

        # Can not use tuple in replicas because tuples are immutable.
        replicas = list(replicas.values())
        members = tuple(self.members.values())
        members_len = len(members)

        if len(replicas) <= members_len:
            for member, replica in zip(members, replicas):
                member.receive_part(replica)
        else:
            while replicas:
                for member, replica in zip(members, replicas):
                    member.receive_part(replica)
                del replicas[:members_len]
    # endregion
