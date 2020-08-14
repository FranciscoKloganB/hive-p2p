"""This module contains domain specific classes that coordinate all
:py:mod:`~domain.cluster_groups` of a simulation instance."""
from __future__ import annotations

import json
import os
from typing import List, Union, Dict, Any

import numpy as np

from domain.cluster_groups import Cluster
from domain.helpers.enums import Status
from domain.helpers.smart_dataclasses import FileBlockData
from domain.network_nodes import HiveNode
from environment_settings import SHARED_ROOT, SIMULATION_ROOT, READ_SIZE, \
    NETWORK_NODES, CLUSTER_GROUPS
from utils.convertions import class_name_to_obj

_PersistentingDict: Dict[str, Dict[str, Union[List[str], str]]]


class Hivemind:
    """Simulation manager class. Plays the role of a master server for all
    Hives of the distributed backup system.

    Class Attributes:
        MAX_EPOCHS:
            The number of time steps a simulation should have (default is 720).
            On a 24 hour day, 720 means one epoch should occur every two minutes.
        MAX_EPOCHS_PLUS_ONE:
            do not alter; (default is MAX_EPOCHS + 1).

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
            A collection of :py:class:`~domain.cluster_groups.Cluster`
            instances managed by the Hivemind.
        network_nodes:
            A dictionary mapping network node identifiers names to their
            object instances (:py:class:`~domain.network_nodes.HiveNode`).
            This collection differs from the
            :py:class:`~domain.cluster_groups.Cluster`s' attribute
            :py:attr:`~domain.cluster_groups.Cluster.members` in the sense that
            the latter is only a subset of `workers`, which includes all
            network nodes of the distributed backup system. Regardless of
            their participation on any Cluster.
    """

    MAX_EPOCHS = None
    MAX_EPOCHS_PLUS_ONE = None

    def __init__(self,
                 simfile_name: str,
                 sid: int,
                 epochs: int,
                 cluster_class: str,
                 node_class: str) -> None:
        """Instantiates an Hivemind object.

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
        Hivemind.MAX_EPOCHS = epochs
        Hivemind.MAX_EPOCHS_PLUS_ONE = epochs + 1

        self.origin = simfile_name
        self.sim_id = sid
        self.epoch = 1

        simfile_path: str = os.path.join(SIMULATION_ROOT, simfile_name)
        with open(simfile_path) as input_file:
            json_obj: Any = json.load(input_file)

            # Init basic simulation variables
            self.cluster_groups: Dict[str, Cluster] = {}
            self.network_nodes: Dict[str, HiveNode] = {}

            # Instantiaite jobless Workers
            for node_id, node_uptime in json_obj['nodes_uptime'].items():
                node = class_name_to_obj(
                    NETWORK_NODES, node_class, [node_id, node_uptime])
                self.network_nodes[node.id] = node

            # Read and split all shareable files specified on the input
            files_spreads: Dict[str, str] = {}
            files_blocks: Dict[str, Dict[int, FileBlockData]] = {}
            blocks: Dict[int, FileBlockData]

            persisting: _PersistentingDict = json_obj['persisting']
            for file_name in persisting:
                with open(os.path.join(SHARED_ROOT, file_name), "rb") as file:
                    part_number: int = 0
                    blocks = {}
                    files_spreads[file_name] = persisting[file_name]['spread']
                    cluster = self.__new_cluster_group(
                        cluster_class, persisting, file_name)
                    while True:
                        read_buffer = file.read(READ_SIZE)
                        if read_buffer:
                            part_number = part_number + 1
                            blocks[part_number] = FileBlockData(
                                cluster.id, file_name, part_number, read_buffer)
                        else:
                            files_blocks[file_name] = blocks
                            break
                    cluster.file.parts_count = part_number

            # Distribute files before starting simulation
            for cluster in self.cluster_groups.values():
                cluster._spread_files(files_spreads[cluster.file.name],
                                      files_blocks[cluster.file.name])

    # endregion

    # region Simulation Interface

    def execute_simulation(self) -> None:
        """Runs a stochastic swarm guidance algorithm applied
        to a P2P network"""
        while self.epoch < Hivemind.MAX_EPOCHS_PLUS_ONE and self.cluster_groups:
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

    # region Keeper Interface

    def receive_complaint(self, suspects_name: str) -> None:
        """Registers a complain against a network node, if enough complaints
        are received, target is evicted from the complainters Cluster.

        Note:
            This method needs implementation at the user descretion.

        Args:
            suspects_name:
                A unique identifier of the suspicious network node.
        """
        # TODO future-iterations:
        #   Move cluster_groups complaint method to this method..
        raise NotImplementedError()

    def find_replacement_node(
            self, exclusion_dict: Dict[str, HiveNode], n: int
    ) -> Dict[str, HiveNode]:
        """Finds a collection of online network nodes that can be used to
        replace offline ones in an Cluster.

        Args:
            exclusion_dict:
                A dictionary of network nodes identifiers and their object
                instances (:py:class:`~domain.network_nodes.HiveNode`),
                which represent the nodes the Cluster is not interested in,
                i.e., this argument is a blacklist.
            n:
                How many replacements the calling Cluster desires to find.

        Returns:
            A collection of replacements which is smaller or equal than `n`.
        """
        selected: Dict[str, HiveNode] = {}
        network_nodes_view = self.network_nodes.copy().values()
        for node in network_nodes_view:
            if len(selected) == n:
                return selected
            elif node.status != Status.ONLINE:
                # TODO: future-iterations review this code.
                self.network_nodes.pop(node.id, None)
            elif node.id not in exclusion_dict:
                selected[node.id] = node
        return selected

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

    # region Helpers

    def __new_cluster_group(
            self, cclass: str, persisting: _PersistentingDict, fname: str
    ) -> Cluster:
        """
        Helper method that initializes a new hive.
        """
        cluster_members: Dict[str, HiveNode] = {}
        size = persisting[fname]['cluster_size']
        nodes = np.random.choice(
            a=[*self.network_nodes.keys()], size=size, replace=False)

        for node_id in nodes:
            cluster_members[node_id] = self.network_nodes[node_id]

        cluster = class_name_to_obj(
            CLUSTER_GROUPS, cclass,
            [self, fname, cluster_members, self.sim_id, self.origin])

        self.cluster_groups[cluster.id] = cluster
        return cluster

    # endregion
