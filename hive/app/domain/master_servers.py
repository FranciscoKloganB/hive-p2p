"""This module contains domain specific classes that coordinate all
:py:mod:`app.domain.cluster_groups` of a simulation instance. These could
simulate centralized authentication servers, file localization or
file metadata servers or a bank of currently online and offline
:py:mod:`storage nodes <app.domain.network_nodes>`."""
from __future__ import annotations

import json

from typing import Union, Dict, Any, Optional

import domain.helpers.enums as e
import type_hints as th
import numpy as np

from utils.convertions import class_name_to_obj
from domain.helpers.smart_dataclasses import FileBlockData
from environment_settings import *

_PersistentingDict: Dict[str, Dict[str, Union[List[str], str]]]


class Master:
    """Simulation manager class, some kind of puppet-master. Could represent
    an authentication server or a monitor that decides along with other
    ``Master`` entities what :py:class:`network nodes
    <app.domain.network_nodes.Node>` are online using consensus algorithms.

    Attributes:
        origin (str):
            The name of the simulation file name that started the simulation
            process.
        sid (int):
            Identifier that generates unique output file names,
            thus guaranteeing that different simulation instances do not
            overwrite previous out files.
        epoch (int):
            The simulation's current epoch.
        cluster_groups (:py:class:`app.type_hints.ClusterDict`)
            A collection of :py:class:`cluster groups
            <app.domain.cluster_groups.Cluster>` managed by the ``Master``.
            Keys are :py:attr:`cluster identifiers
            <app.domain.cluster_groups.Cluster.id>` and values are the
            cluster instances.
        network_nodes (:py:class:`app.type_hints.NodeDict`):
            A dictionary mapping :py:attr:`node identifiers
            <app.domain.network_nodes.Node.id>` to their instance objects.
            This collection differs from
            :py:attr:`app.domain.cluster_groups.Cluster.members` attribute
            in the sense that the former ``network_nodes`` includes all
            nodes, both online and offline, available on the entire
            distributed backup storage system regardless of their
            participation in any :py:class:`cluster group
            <app.domain.cluster_groups.Cluster>`.
    """

    MAX_EPOCHS: Optional[int] = None
    MAX_EPOCHS_PLUS_ONE: Optional[int] = None

    def __init__(self,
                 simfile_name: str,
                 sid: int,
                 epochs: int,
                 cluster_class: str,
                 node_class: str) -> None:
        """Instantiates an Master object.

        Args:
            simfile_name:
                A path to the simulation file to be run by the simulator.
            sid:
                Identifier that generates unique output file names,
                thus guaranteeing that different simulation instances do not
                overwrite previous out files.
            epochs:
                The number of discrete time steps the simulation lasts.
            cluster_class:
                The name of the class used to instantiate cluster group
                instances through reflection. See :py:mod:`cluster groups module
                <app.domain.cluster_groups>`.
            node_class:
                The name of the class used to instantiate network node
                instances through reflection. See :py:mod:`network nodes module
                <app.domain.network_nodes>`.
        """
        Master.MAX_EPOCHS = epochs
        Master.MAX_EPOCHS_PLUS_ONE = epochs + 1

        self.origin = simfile_name
        self.sim_id = sid
        self.epoch = 1
        self.cluster_groups: th.ClusterDict = {}
        self.network_nodes: th.NodeDict = {}

        simfile_path: str = os.path.join(SIMULATION_ROOT, simfile_name)
        self._process_simfile(simfile_path, cluster_class, node_class)

    # region Simulation setup
    def _process_simfile(
            self, path: str, cluster_class: str, node_class: str) -> None:
        """Opens and processes the simulation filed referenced in ``path``.

        This method opens the file reads the json data inside it. Combined
        with :py:mod:`app.environment_settings` it sets up the class
        instances to be used during the simulation (e.g.,
        :py:class:`cluster groups <app.domain.cluster_groups.Cluster>` and
        :py:class:`network nodes <app.domain.network_nodes.Node>`). This
        method also be splits the file to be persisted in the simulation into
        multiple ``blocks`` or ``chunks`` and for triggering the initial
        :py:meth:`file spreading
        <app.domain.cluster_groups.Cluster.spread_files>` mechanism.

        Args:
            path:
                The path to the simulation file. Including extension and
                parent folders.
            cluster_class:
                The name of the class used to instantiate cluster group
                instances through reflection.
                See :py:mod:`app.domain.cluster_groups`.
            node_class:
                The name of the class used to instantiate network node
                instances through reflection.
                See :py:mod:`app.domain.network_nodes`.
        """
        with open(path) as input_file:
            simfile_json: Any = json.load(input_file)

            fspreads: Dict[str, str] = {}
            fblocks: Dict[str, th.ReplicasDict] = {}

            self._create_network_nodes(simfile_json, node_class)

            d: _PersistentingDict = simfile_json['persisting']
            for fname in d:
                spread_strategy = d[fname]['spread']
                fspreads[fname] = spread_strategy
                size = d[fname]['cluster_size']
                cluster = self._new_cluster_group(cluster_class, size, fname)
                fblocks[fname] = self._split_files(fname, cluster, READ_SIZE)

            # Distribute files before starting simulation
            for cluster in self.cluster_groups.values():
                spread_strategy = fspreads[cluster.file.name]
                file_blocks = fblocks[cluster.file.name]
                cluster.spread_files(file_blocks, spread_strategy)

    def _create_network_nodes(
            self, json: Dict[str, Any], node_class: str) -> None:
        """Helper method that instantiates all
        :py:class:`network nodes <app.domain.network_nodes.Node>` that are
        specified in the simulation file.

        Args:
            json:
                The simulation file in JSON dictionary object format.
            node_class:
                The type of network node to create.
        """
        for nid, nuptime in json['nodes_uptime'].items():
            node = self._new_network_node(node_class, nid, nuptime)
            self.network_nodes[nid] = node

    def _split_files(
            self, fname: str, cluster: th.ClusterType, bsize: int
    ) -> th.ReplicasDict:
        """Helper method that splits the files into multiple blocks to be
        persisted in a :py:class:`cluster group
        <app.domain.cluster_groups.Cluster>`.

        Args:
            fname:
                The name of the file located in
                :py:const:`~app.environment_settings.SHARED_ROOT` folder to be
                read and splitted.
            cluster (:py:class:`~app.type_hints.ClusterType`):
                A reference to the :py:class:`cluster group
                <app.domain.cluster_groups.Cluster>` whose
                :py:attr:`~app.domain.cluster_groups.Cluster.members` will be
                responsible for ensuring the file specified in ``fname``
                becomes durable.
            bsize:
                The maximum amount of bytes each file block can have.

        Returns:
            :py:class:`~app.type_hints.ReplicasDict`:
                A dictionary in which the keys are integers and values are
                :py:class:`file blocks
                <app.domain.helpers.smart_dataclasses.FileBlockData>`, whose
                attribute :py:attr:`~app.domain.helpers.smart_dataclasses.FileBlockData.number`
                is the key.
        """
        with open(os.path.join(SHARED_ROOT, fname), "rb") as file:
            bid: int = 0
            d: th.ReplicasDict = {}
            while True:
                read_buffer = file.read(bsize)
                if read_buffer:
                    bid += 1
                    d[bid] = FileBlockData(cluster.id, fname, bid, read_buffer)
                else:
                    break
            cluster.file.parts_count = bid
            return d
    # endregion

    # region Simulation steps
    def execute_simulation(self) -> None:
        """Starts the simulation processes."""
        while self.epoch < Master.MAX_EPOCHS_PLUS_ONE and self.cluster_groups:
            print("epoch: {}".format(self.epoch))
            terminated_clusters: List[str] = []
            for cluster in self.cluster_groups.values():
                cluster.execute_epoch(self.epoch)
                if not cluster.running:
                    terminated_clusters.append(cluster.id)
                    cluster.file.jwrite(cluster, self.origin, self.epoch)
            for cid in terminated_clusters:
                print(f"Cluster: {cid} terminated at epoch {self.epoch}")
                self.cluster_groups.pop(cid)
            self.epoch += 1
    # endregion

    # region Master API
    def find_online_nodes(
            self, n: int = 1, blacklist: Optional[th.NodeDict] = None
    ) -> th.NodeDict:
        """Finds ``n`` :py:class:`network nodes
        <app.domain.network_nodes.Node>` who are currently registered at the
        ``Master`` and whose status is online.

        Args:
            n:
                How many :py:class:`network node
                <app.domain.network_nodes.Node>` references the requesting
                entity wants to find.
            blacklist (:py:class:`~app.type_hints.NodeDict`):
                A collection of :py:attr:`nodes identifiers
                <app.domain.network_nodes.Node.id>` and their object
                instances, which specify nodes the requesting entity has
                no interest in.

        Returns:
            :py:class:`~app.type_hints.NodeDict`:
                A collection of :py:class:`network nodes <app.domain.network_nodes.Node>`
                which is at most as big as ``n``, which does not include any
                node named in ``blacklist``.
        """

        selected: th.NodeDict = {}
        if n < 1:
            return selected
        if blacklist is None:
            blacklist = {}

        network_nodes_view = self.network_nodes.copy().values()
        for node in network_nodes_view:
            if len(selected) >= n:
                return selected
            if node.id not in blacklist:
                selected[node.id] = node
        return selected
    # endregion

    # region Helpers
    def _new_cluster_group(
            self, cluster_class: str, size: int, fname: str
    ) -> th.ClusterType:
        """Helper method that initializes a new Cluster group.

        Args:
            cluster_class:
                The name of the class used to instantiate cluster group
                instances through reflection. See :py:mod:`cluster groups module
                <app.domain.cluster_groups>`.
            size:
                The :py:class:`cluster's <app.domain.cluster_groups.Cluster>`
                initial memberhip size.
            fname:
                The name of the fille being stored in the cluster.

        Returns:
            :py:class:`~app.type_hints.ClusterType`:
                The :py:class:`~app.domain.cluster_groups.Cluster` instance.
        """
        cluster_members: th.NodeDict = {}
        nodes = np.random.choice(
            a=tuple(self.network_nodes), size=size, replace=False)

        for node_id in nodes:
            cluster_members[node_id] = self.network_nodes[node_id]

        cluster = class_name_to_obj(
            CLUSTER_GROUPS,
            cluster_class,
            [self, fname, cluster_members, self.sim_id, self.origin]
        )

        self.cluster_groups[cluster.id] = cluster
        return cluster

    def _new_network_node(
            self, node_class: str, nid: str, node_uptime: str) -> th.NodeType:
        """Helper method that initializes a new Node.

        Args:
            node_class:
                The name of the class used to instantiate network node
                instances through reflection. See :py:mod:`network nodes module
                <app.domain.network_nodes>`.
            nid:
                An id that will uniquely identifies the
                :py:class:`network node <app.domain.network_nodes.Node>`.
            node_uptime:
                A float value in string representation that defines the
                uptime of the network node.

        Returns:
            :py:class:`~app.type_hints.NodeType`:
                The :py:class:`~app.domain.network_nodes.Node` instance.
        """
        return class_name_to_obj(NETWORK_NODES, node_class, [nid, node_uptime])
    # endregion


class HiveMaster(Master):
    # region Master API
    def get_cloud_reference(self) -> str:
        """Use to obtain a reference to 3rd party cloud storage provider

        The cloud storage provider can be used to temporarely host files
        belonging to :py:class:`cluster clusters <app.domain.HiveCluster>` in bad
        conditions that may compromise the file durability of the files they
        are responsible for persisting.

        Note:
            This method is virtual.

        Returns:
            A pointer to the cloud server, e.g., an IP Address.
        """
        return ""
    # endregion


class HDFSMaster(Master):
    # region Simulation setup
    def _process_simfile(
            self, path: str, cluster_class: str, node_class: str) -> None:
        """Opens and processes the simulation filed referenced in `path`.

        Overrides:
            :py:meth:`app.domain.master_servers.Master._process_simfile`.

            The method is exactly the same except for one instruction. The
            :py:meth:`~app.domain.master_servers.Master._split_files` is
            invoked with fixed `bsize` = 1MB. The reason for this is
            two-fold:

                - The default and, thus recommended, block size for the \
                hadoop distributed file system is 128MB. The system is not \
                designed to perform well with small file blocks, but Hives \
                requires many file blocks for stochastic swarm guidance to \
                work, hence being more effective with small block sizes. By \
                default Hives runs with 128KB blocks.

                - Hadoop limits the minimum block size to be 1MB, \
                `dfs.namenode.fs-limits.min-block-size <https://hadoop.apache.org/docs/r2.6.0/hadoop-project-dist/hadoop-hdfs/hdfs-default.xml#dfs.namenode.fs-limits.min-block-size>`_. \
                For this reason, we make HDFSMaster split files into 1MB \
                chunks, as that is the closest we would get to our Hive's \
                default block size in the real world.

            The other difference is that the spread strategy is ignored.
            We are not interested in knowing if the way the files are
            initially spread affects the time it takes for clusters to
            achieve a steady-state distribution since in HDFS
            :py:class:`file block replicas
            <app.domain.helpers.smart_dataclasses.FileBlockData>` are
            stationary on data nodes until they die.

        Args:
            path:
                The path to the simulation file. Including extension and
                parent folders.
            cluster_class:
                The name of the class used to instantiate cluster group
                instances through reflection.
                See :py:mod:`app.domain.cluster_groups`.
            node_class:
                The name of the class used to instantiate network node
                instances through reflection.
                See :py:mod:`app.domain.network_nodes`.
        """
        with open(path) as input_file:
            simfile_json: Any = json.load(input_file)

            fspreads: Dict[str, str] = {}
            fblocks: Dict[str, th.ReplicasDict] = {}

            self._create_network_nodes(simfile_json, node_class)

            d: _PersistentingDict = simfile_json['persisting']
            for fname in d:
                spread_strategy = d[fname]['spread']
                fspreads[fname] = spread_strategy
                size = d[fname]['cluster_size']
                cluster = self._new_cluster_group(cluster_class, size, fname)
                fblocks[fname] = self._split_files(fname, cluster, 1048576)

            # Distribute files before starting simulation
            for cluster in self.cluster_groups.values():
                file_blocks = fblocks[cluster.file.name]
                cluster.spread_files(file_blocks)
    # endregion


class NewscastMaster(Master):
    def __init__(self,
                 simfile_name: str,
                 sid: int,
                 epochs: int,
                 cluster_class: str,
                 node_class: str) -> None:
        super().__init__(simfile_name, sid, epochs, cluster_class, node_class)
        for cluster in self.cluster_groups.values():
            cluster.wire_k_out()

    # region Simulation setup
    def _process_simfile(
            self, path: str, cluster_class: str, node_class: str) -> None:
        """Opens and processes the simulation filed referenced in `path`.

        Overrides:
            :py:meth:`app.domain.master_servers.Master._process_simfile`.

            Newscast is a gossip-based P2P network. We assume erasure-coding
            would be used in this scenario and thus, for simplicity,
            we divide the specified file's size into multiple ``1/N``,
            where ``N`` is the number of :py:class:`network nodes
            <app.domain.network_nodes.NewscastNode>` in the system.

        Note:
            This class, :py:class:`~app.domain.cluster_groups.NewscastCluster`
            and :py:class:`~app.domain.network_nodes.NewscastNode` were
            created to test our simulators performance, concerning the amount
            of supported simultaneous network nodes in a simulation. We do
            not actually care if the created file blocks are lost as the
            :py:class:`network nodes <app.domain.network_nodes.NewscastNode>`
            job in the simulation is to carry out the
            protocol defined in `PeerSim's AverageFunction
            <http://peersim.sourceforge.net/doc/index.html>`_. `PeerSim
            <http://peersim.sourceforge.net/>`_ uses configuration ``Example 2``
            provided in release 1.0.5, as a means of testing the simulator
            performance, according to this `Ms.C. dissertation by J. Neto
            <https://www.gsd.inesc-id.pt/~lveiga/papers/msc-supervised-thesis-abstracts/jneto-FINAL.pdf>`_.
            This configuration uses Newscast protocol with AverageFunction
            and periodic monitoring of the system state. We implement our
            version of `Adaptaive Peer Sampling with Newscast
            <https://dl.acm.org/doi/abs/10.1007/978-3-642-03869-3_50>`_ by
            N. Tölgyesi and M. Jelasity, to avoid the effort of translating
            PeerSim's code.

        Args:
            path:
                The path to the simulation file. Including extension and
                parent folders.
            cluster_class:
                The name of the class used to instantiate cluster group
                instances through reflection.
                See :py:mod:`app.domain.cluster_groups`.
            node_class:
                The name of the class used to instantiate network node
                instances through reflection.
                See :py:mod:`app.domain.network_nodes`.
        """
        with open(path) as input_file:
            simfile_json: Any = json.load(input_file)

            self._create_network_nodes(simfile_json, node_class)

            d: _PersistentingDict = simfile_json['persisting']
            for fname in d:
                spread_strategy = d[fname]['spread']
                cluster_size = d[fname]['cluster_size']

                cluster = self._new_cluster_group(
                    cluster_class, cluster_size, fname)

                file_path = os.path.join(SHARED_ROOT, fname)
                block_size = os.path.getsize(file_path) / cluster_size
                file_blocks = self._split_files(fname, cluster, block_size)

                cluster.spread_files(file_blocks, spread_strategy)
    # endregion
