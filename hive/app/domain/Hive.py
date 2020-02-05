import math
import uuid
from typing import Dict, List, Any

import numpy as np
import pandas as pd

import utils.matrices as matrices
import utils.metropolis_hastings as mh
from domain.Enums import Status, HttpCodes
from domain.Hivemind import Hivemind
from domain.SharedFilePart import SharedFilePart
from domain.Worker import Worker
from domain.helpers.FileData import FileData
from globals.globals import REPLICATION_LEVEL, DEFAULT_COLUMN


class Hive:
    """
    :ivar str id: unique identifier in str format
    :ivar Hivemind hivemind: reference to the master server, which in this case is just a simulator program
    :ivar FileData Union[None, FileData]: instance of class FileData which contains information regarding the file persisted by this hive
    :ivar Dict[str, Worker] members: Workers that belong to this P2P Hive, key is worker.id, value is the respective Worker instance
    :ivar int hive_size: integer that keeps track of members collection size
    :ivar int critical_size: minimum number of replicas required for data recovery plus the number of peer faults the system must support during replication.
    :ivar int sufficient_size: depends on churn-rate and equals critical_size plus the number of peers expected to fail between two successive recovery phases
    :ivar int redudant_size: application-specific system parameter, but basically represents that the hive is to big
    :ivar DataFrame desired_distribution: distribution hive members are seeking to achieve for each the files they persist together.
    """
    # region Class Variables, Instance Variables and Constructors
    def __init__(self, hivemind: Hivemind, file_name: str, members: Dict[str, Worker]) -> None:
        """
        Instantiates an Hive abstraction
        :param Hivemind hivemind: Hivemand instance object which leads the simulation
        :param str file_name: name of the file this Hive is responsible for
        :param Dict[str, Worker] members: collection mapping names of the Hive's initial workers' to their Worker instances
        """
        self.id: str = str(uuid.uuid4())
        self.hivemind = hivemind
        self.file: FileData = FileData(file_name)
        self.members: Dict[str, Worker] = members
        self.original_size: int = len(members)
        self.hive_size: int = len(members)
        self.critical_size: int = REPLICATION_LEVEL
        self.sufficient_size: int = self.critical_size + math.ceil(self.hive_size * 0.34)
        self.redudant_size: int = self.sufficient_size + self.hive_size
        self.desired_distribution = None
        self.broadcast_transition_matrix(self.new_transition_matrix())  # implicitly inits self.desired_distribution within new_transition_matrix()

    # endregion

    # region Routing
    def route_to_cloud(self) -> None:
        """
        TODO: future-iterations
        Remaining hive members upload all data they have to a cloud server
        """
        # noinspection PyUnusedLocal
        cloud_ref: str = self.hivemind.get_cloud_reference()

    def route_part(self, destination_name: str, part: SharedFilePart) -> Any:
        """
        Receives a shared file part and sends it to the given destination
        :param str destination_name: destination worker's id
        :param SharedFilePart part: the file part to send to specified worker
        :returns int: http codes based status of destination worker
        """
        member = self.members[destination_name]
        if member.status == Status.ONLINE:
            member.receive_part(part)
            return HttpCodes.OK
        else:
            return HttpCodes.NOT_FOUND
    # endregion

    # region Swarm Guidance
    def new_desired_distribution(self, member_ids: List[str], member_uptimes: List[float]) -> List[float]:
        """
        Normalizes inputted member uptimes and saves it on Hive.desired_distribution attribute
        :param List[str] member_ids: list of member ids representing the current hive membership
        :param List[float] member_uptimes: list of member uptimes to be normalized
        :returns List[float] desired_distribution: uptimes represent 'reliability', thus, desired distribution is the normalization of the members' uptimes
        """
        uptime_sum = sum(member_uptimes)
        uptimes_normalized = [member_uptime / uptime_sum for member_uptime in member_uptimes]

        self.desired_distribution = pd.DataFrame(data=uptimes_normalized, index=member_ids)
        self.file.desired_distribution = self.desired_distribution
        self.file.current_distribution = pd.DataFrame(data=[0] * len(uptimes_normalized), index=member_ids)

        return uptimes_normalized

    def new_transition_matrix(self) -> pd.DataFrame:
        """
        returns DataFrame: Creates a new transition matrix for the members of the Hive, to be followed independently by each of them
        """
        desired_distribution: List[float]
        adjancency_matrix: List[List[int]]
        member_uptimes: List[float] = []
        member_ids: List[str] = []

        for worker in self.members.values():
            member_uptimes.append(worker.uptime)
            member_ids.append(worker.id)

        adjancency_matrix = matrices.new_symmetric_adjency_matrix(len(member_ids))
        desired_distribution = self.new_desired_distribution(member_ids, member_uptimes)

        transition_matrix: np.ndarray = mh.metropolis_algorithm(adjancency_matrix, desired_distribution, column_major_out=True)
        return pd.DataFrame(transition_matrix, index=member_ids, columns=member_ids)

    def broadcast_transition_matrix(self, transition_matrix: pd.DataFrame) -> None:
        """
        Gives each member his respective slice (vector column) of the transition matrix the Hive is currently executing.
        post-scriptum: we could make an optimization that sets a transition matrix for the hive, ignoring the file names, instead of mapping different file
        names to an equal transition matrix within each hive member, thus reducing space overhead arbitrarly, however, this would make Simulation harder. This
        note is kept for future reference. This also assumes an hive can store multiple files. For simplicity each Hive only manages one file for now.
        """
        transition_vector: pd.DataFrame
        for worker in self.members.values():
            transition_vector = transition_matrix.loc[:, worker.id]
            worker.set_file_routing(self.file.name, transition_vector)
    # endregion

    # region Simulation Interface
    def spread_files(self, spread_mode: str, file_parts: Dict[int, SharedFilePart]):
        """
        Spreads files over the initial members of the Hive
        :param str spread_mode: 'u' for uniform distribution, 'a' one peer receives all or 'i' to distribute according to the desired steady state distribution
        :param Dict[int, SharedFilePart] file_parts: file parts to distribute over the members
        """
        if spread_mode == "a":
            choices: List[Worker] = [*self.members.values()]
            workers: List[Worker] = np.random.choice(a=choices, size=REPLICATION_LEVEL, replace=False)
            for worker in workers:
                for part in file_parts.values():
                    worker.receive_part(part)

        elif spread_mode == "u":
            for part in file_parts.values():
                choices: List[Worker] = [*self.members.values()]
                workers: List[Worker] = np.random.choice(a=choices, size=REPLICATION_LEVEL, replace=False)
                for worker in workers:
                    worker.receive_part(part)

        elif spread_mode == 'i':
            choices = [*self.members.values()]
            desired_distribution: List[float] = []
            for member_id in choices:
                desired_distribution.append(self.desired_distribution.loc[member_id, DEFAULT_COLUMN].item())

            for part in file_parts.values():
                choices: List[Worker] = choices.copy()
                workers: List[Worker] = np.random.choice(a=choices, p=desired_distribution, size=REPLICATION_LEVEL, replace=False)
                for worker in workers:
                    worker.receive_part(part)

    def execute_epoch(self) -> bool:
        """
        Orders all members to execute their epoch, i.e., perform stochastic swarm guidance for every file they hold
        """
        recoverable_parts: Dict[int, SharedFilePart] = {}
        disconnected_workers: List[Worker] = []

        # Members execute epoch
        for worker in self.members.values():
            if worker.get_epoch_status() == Status.OFFLINE:  # Offline members require possible changes to membership and file maintenance
                disconnected_workers.append(worker)
                lost_parts: Dict[int, SharedFilePart] = worker.get_file_parts(self.file.name)
                # Process data held by the disconnected worker
                for number, part in lost_parts.items():
                    if part.decrease_and_get_references() > 0:
                        recoverable_parts[number] = part
                    else:
                        recoverable_parts.pop(number)  # this pop isn't necessary, but remains here for piece of mind and explicit explanation, O(1) anyway
                        return False  # This release only uses replication, thus having 0 references makes it impossible to recover original file
            else:  # Member is still online this epoch, so he can execute his own part of the epoch
                worker.execute_epoch(self, self.file.name)

        # Perfect failure detection, assumes that once a machine goes offline it does so permanently for all hives, so, pop members who disconnected
        if len(disconnected_workers) == self.hive_size:
            return False  # Hive is completly offline, simulation failed
        elif len(disconnected_workers) > 0:
            self.membership_maintenance(disconnected_workers)

        for part in recoverable_parts.values():
            part.set_epochs_to_recover()
    # endregion

    # region Helpers
    def membership_maintenance(self, disconnected_workers: List[Worker]) -> None:
        """
        Used to ensure hive stability and proper swarm guidance behavior. No maintenance is needed if there are no disconnected workers in the inputed list.
        :param List[Worker] disconnected_workers: collection of members who disconnected during this epoch
        """
        for member in disconnected_workers:
            self.members.pop(member.id)

        self.hive_size = len(self.members)

        if self.hive_size < self.critical_size:
            self.route_to_cloud()

        if self.hive_size < self.sufficient_size:
            new_members: Dict[str, Worker] = self.hivemind.find_replacement_worker(self.members, self.original_size - self.hive_size)
            if not new_members:
                return
            self.members.update(new_members)
            self.hive_size = len(self.members)
        elif self.hive_size > self.redudant_size:
            # TODO: future-iterations
            #  evict peers
            self.hive_size = len(self.members)

        self.broadcast_transition_matrix(self.new_transition_matrix())
    # endregion
