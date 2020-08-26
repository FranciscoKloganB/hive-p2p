"""This module contains domain specific classes that coordinate all
:py:mod:`domain.cluster_groups` of a simulation instance."""
from __future__ import annotations

import json
from typing import List, Union, Dict, Any, Optional

import domain.helpers.enums as e
import type_hints as th
import numpy as np

from utils.convertions import class_name_to_obj
from domain.helpers.smart_dataclasses import FileBlockData
from environment_settings import *

_PersistentingDict: Dict[str, Dict[str, Union[List[str], str]]]


class Master:
    """Simulation manager class. Plays the role of a master server for all
    Hives of the distributed backup system.

    Attributes:
        origin:
            The name of the simulation file name that started the simulation
            process.
        sid:
            Identifier that generates unique output file names,
            thus guaranteeing that different simulation instances do not
            overwrite previous out files.
        epoch:
            The simulation's current epoch.
        cluster_groups:
            A collection of :py:class:`domain.cluster_groups.Cluster`
            instances managed by the Master.
        network_nodes:
            A dictionary mapping network node identifiers names to their
            object instances (:py:class:`domain.network_nodes.HiveNode`).
            This collection differs from the
            :py:class:`Cluster's <app.domain.cluster_groups.Cluster>` attribute
            :py:attr:`~app.domain.cluster_groups.Cluster.members` in the sense
            that the latter is only a subset of `workers`, which includes all
            network nodes of the distributed backup system, regardless of
            their participation on any Cluster.
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
                instances through reflection. See :py:mod:`Cluster Group
                <domain.cluster_groups>`.
            node_class:
                The name of the class used to instantiate network node
                instances through reflection. See :py:mod:`Network Node
                <domain.network_nodes>`.
        """
        Master.MAX_EPOCHS = epochs
        Master.MAX_EPOCHS_PLUS_ONE = epochs + 1

        self.origin = simfile_name
        self.sim_id = sid
        self.epoch = 1
        self.cluster_groups: th.ClusterDict = {}
        self.network_nodes: th.NodeDict = {}

        simfile_path: str = os.path.join(SIMULATION_ROOT, simfile_name)
        self.__process_simfile__(simfile_path, cluster_class, node_class)

    # region Simulation setup
    def __process_simfile__(
            self, path: str, cluster_class: str, node_class: str) -> None:
        """Opens and processes the simulation filed referenced in `path`.

        This method opens the file reads the json data inside it and combined
        with :py:mod:`environment_settings`, sets up the class instances to
        be used during the simulation (e.g., :py:class:`Clusters
        <domain.network_nodes.Cluster>` and :py:class:`Nodes
        <domain.network_nodes.Node>`). This method should also be responsible
        for splitting the file into multiple chunks/blocks/parts and
        distributing them over the initial clusters'
        :py:attr:`domain.cluster_groups.Cluster.members`.

        Args:
            path:
                The path to the simulation file. Including extension and
                parent folders.
            cluster_class:
                The name of the class used to instantiate cluster group
                instances through reflection. See :py:mod:`Cluster Group
                <domain.cluster_groups>`.
            node_class:
                The name of the class used to instantiate network node
                instances through reflection. See :py:mod:`Network Node
                <domain.network_nodes>`.
        """
        with open(path) as input_file:
            simfile_json: Any = json.load(input_file)

            fspreads: Dict[str, str] = {}
            fblocks: Dict[str, th.ReplicasDict] = {}

            self.__create_network_nodes__(simfile_json, node_class)

            d: _PersistentingDict = simfile_json['persisting']
            for fname in d:
                spread_strategy = d[fname]['spread']
                fspreads[fname] = spread_strategy
                size = d[fname]['cluster_size']
                cluster = self.__new_cluster_group__(cluster_class, size, fname)
                fblocks[fname] = self.__split_files__(fname, cluster, READ_SIZE)

            # Distribute files before starting simulation
            for cluster in self.cluster_groups.values():
                spread_strategy = fspreads[cluster.file.name]
                file_blocks = fblocks[cluster.file.name]
                cluster.spread_files(file_blocks, spread_strategy)

    def __create_network_nodes__(
            self, json: Dict[str, Any], node_class: str) -> None:
        """Helper method that instantiates all
        :py:class:`Network Nodes<domain.network_nodes.Node> that are
        specified in the simulation file.

        Args:
            json:
                The simulation file in JSON dictionary object format.
            node_class:
                The type of network node to create.
        """
        for nid, nuptime in json['nodes_uptime'].items():
            node = self.__new_network_node__(node_class, nid, nuptime)
            self.network_nodes[nid] = node

    def __split_files__(
            self, fname: str, cluster: th.ClusterType, bsize: int
    ) -> th.ReplicasDict:
        """Helper method that splits the files into multiple blocks to be
        persisted in a :py:class:`domain.cluster_group.Cluster`.

        Args:
            fname:
                The name of the file located in
                :py:const:`environment_settings.SHARED_ROOT` folder to be
                read and splitted.
            bsize:
                The maximum amount of bytes each file block can have.

        Returns:

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
        """Runs a stochastic swarm guidance algorithm applied
        to a P2P network.
        """
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
    def find_replacement_node(
            self, exclusion_dict: th.NodeDict, n: int) -> th.NodeDict:
        """Finds a collection of online network nodes that can be used to
        replace offline ones in an Cluster.

        Args:
            exclusion_dict:
                A dictionary of network nodes identifiers and their object
                instances (:py:class:`app.domain.network_nodes.HiveNode`),
                which represent the nodes the Cluster is not interested in,
                i.e., this argument is a blacklist.
            n:
                How many replacements the calling Cluster desires to find.

        Returns:
            A collection of replacements which is smaller or equal than `n`.
        """
        selected: th.NodeDict = {}
        if n <= 0:
            return selected

        network_nodes_view = self.network_nodes.copy().values()
        for node in network_nodes_view:
            if len(selected) == n:
                return selected
            elif node.status != e.Status.ONLINE:
                # TODO: future-iterations review this code.
                self.network_nodes.pop(node.id, None)
            elif node.id not in exclusion_dict:
                selected[node.id] = node
        return selected
    # endregion

    # region Helpers
    def __new_cluster_group__(
            self, cluster_class: str, size: int, fname: str
    ) -> th.ClusterType:
        """Helper method that initializes a new Cluster group.

        Args:
            cluster_class:
                The name of the class used to instantiate cluster group
                instances through reflection. See :py:mod:`Cluster Group
                <domain.cluster_groups>`.
            size:
                The :py:class:`Cluster's <app.domain.cluster_groups.Cluster>`
                initial member size.
            fname:
                The name of the fille being stored in the cluster.

        Returns:
            The :py:class:`~app.domain.cluster_groups.Cluster` instance.
        """
        cluster_members: th.NodeDict = {}
        nodes = np.random.choice(
            a=[*self.network_nodes.keys()], size=size, replace=False)

        for node_id in nodes:
            cluster_members[node_id] = self.network_nodes[node_id]

        cluster = class_name_to_obj(
            CLUSTER_GROUPS,
            cluster_class,
            [self, fname, cluster_members, self.sim_id, self.origin]
        )

        self.cluster_groups[cluster.id] = cluster
        return cluster

    def __new_network_node__(
            self, node_class: str, nid: str, node_uptime: str) -> th.NodeType:
        """Helper method that initializes a new Node.

        Args:
            node_class:
                The name of the class used to instantiate network node
                instances through reflection. See :py:mod:`Network Node
                <domain.network_nodes>`.
            nid:
                An id that will uniquely identifies the network node.
            node_uptime:
                A float value in string representation that defines the
                uptime of the network node.

        Returns:
            The :py:class:`app.domain.network_nodes.Node` instance.
        """
        return class_name_to_obj(NETWORK_NODES, node_class, [nid, node_uptime])
    # endregion


class HiveMaster(Master):
    def __init__(self,
                 simfile_name: str,
                 sid: int,
                 epochs: int,
                 cluster_class: str,
                 node_class: str) -> None:
        super().__init__(simfile_name, sid, epochs, cluster_class, node_class)

    # region Master API
    def get_cloud_reference(self) -> str:
        """Use to obtain a reference to 3rd party cloud storage provider

        The cloud storage provider can be used to temporarely host files
        belonging to Hives in bad status, thus increasing file durability
        in the system.

        Note:
            TODO: This method requires implementation at the user descretion.

        Returns:
            A pointer to thhe cloud server, e.g., an IP Address.
        """
        return ""
    # endregion


class HDFSMaster(Master):
    def __init__(self,
                 simfile_name: str,
                 sid: int,
                 epochs: int,
                 cluster_class: str,
                 node_class: str) -> None:
        super().__init__(simfile_name, sid, epochs, cluster_class, node_class)

    # region Simulation setup
    def __process_simfile__(
            self, path: str, cluster_class: str, node_class: str) -> None:
        """Opens and processes the simulation filed referenced in `path`.

        Overrides:
            py:mod:`app.domain.master_servers.Master.__process_simfile__`. The
            method is exactly the same except for one instruction. The
            :py:mod:`app.domain.master_servers.Master.__split_files__` is
            invoked with fixed `bsize` = 1MB. The reason for this is twofold::

                - The default and, thus recommended, block/chunk size for the
                hadoop distributed file system is 128MB. The system is not
                designed to perform well with small file blocks, but Hives
                requires many file blocks for stochastic swarm guidance to
                work, hence being more effective with small block sizes. By
                default Hives runs with 128KB blocks.
                - Hadoop limits the minimum block size to be 1MB,
                `dfs.namenode.fs-limits.min-block-size
                <https://hadoop.apache.org/docs/r2.6.0/hadoop-project-dist/hadoop-hdfs/hdfs-default.xml#dfs.namenode.fs-limits.min-block-size>`.
                For this reason, we make HDFSMaster split files into 1MB chunks,
                as that is the closest we would get to our Hive's default
                block size in the real wourld

            The other difference, is that spread strategy is ignored as we
            are not interested in knowing if the way the files are initially
            spread affect the time the time it takes for hives to achieve a
            steady state distribution, since in HDFS file block replicas are
            stationary on data nodes until they die.
        """
        with open(path) as input_file:
            simfile_json: Any = json.load(input_file)

            fspreads: Dict[str, str] = {}
            fblocks: Dict[str, th.ReplicasDict] = {}

            self.__create_network_nodes__(simfile_json, node_class)

            d: _PersistentingDict = simfile_json['persisting']
            for fname in d:
                spread_strategy = d[fname]['spread']
                fspreads[fname] = spread_strategy
                size = d[fname]['cluster_size']
                cluster = self.__new_cluster_group__(cluster_class, size, fname)
                fblocks[fname] = self.__split_files__(fname, cluster, 1048576)

            # Distribute files before starting simulation
            for cluster in self.cluster_groups.values():
                file_blocks = fblocks[cluster.file.name]
                cluster.spread_files(file_blocks)
    # endregion
