import uuid
import numpy as np
import pandas as pd
import utils.matrices as matrices
import utils.metropolis_hastings as mh

from domain.Worker import Worker
from domain.Enums import Status, HttpCodes
from domain.helpers.FileData import FileData
from domain.SharedFilePart import SharedFilePart

from globals.globals import REPLICATION_LEVEL, DEFAULT_COLUMN

from typing import Dict, List, Union, Any


class Hive:
    """
    :ivar str id: unique identifier in str format
    :ivar Dict[str, Worker] members: Workers that belong to this P2P Hive, key is worker.id, value is the respective Worker instance
    :ivar FileData Union[None, FileData]: instance of class FileData which contains information regarding the file persisted by this hive
    :ivar DataFrame desired_distribution: distribution hive members are seeking to achieve for each the files they persist together.
    :deprecated_ivar Dict[str, FileData] files: maps the name of the files persisted by the members of this Hive, to instances of FileData used by the Simulator class
    """
    # region Class Variables, Instance Variables and Constructors
    def __init__(self, members: Dict[str, Worker]) -> None:
        """
        Instantiates an Hive abstraction
        :param Dict[str, Worker] members: collection mapping names of the Hive's initial workers' to their Worker instances
        """
        self.id: str = str(uuid.uuid4())
        self.members: Dict[str, Worker] = members
        self.file: Union[None, FileData] = None

        transition_matrix: pd.DataFrame = self.new_transition_matrix()
        self.broadcast_transition_matrix(transition_matrix)

        self.desired_distribution: pd.DataFrame = pd.DataFrame()
        # self.files: Dict[str, FileData] = {}

    # endregion

    # region Routing
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
        uptimes_normalized = [member_uptime/uptime_sum for member_uptime in member_uptimes]
        self.desired_distribution = pd.DataFrame(data=uptimes_normalized, index=member_ids)
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

    # region Membership
    def remove_member(self, worker: Union[str, Worker]) -> None:
        """
        Removes a worker from the Hive's membership set.
        :param Union[str, Worker] worker: id of the worker or instance object of class Worker to be removed from the set.
        """
        self.members.pop(worker, None)

    def add_member(self, worker: Worker) -> None:
        """
        Adds a worker to the Hive's membership set.
        :param Worker worker: instance object of class Worker to be added to the set.
        """
        self.members[worker.id] = worker

    def replace_member(self, old_member: Union[str, Worker], new_member: Worker) -> None:
        """
        Replaces a worker from the Hive's membership set with some other worker.
        :param Union[str, Worker] old_member: id of the worker or instance object of class Worker to be replaced in the set.
        :param Worker new_member: instance object of class Worker to be added to the set.
        """
        self.members.pop(old_member)
        self.members[new_member.id] = new_member
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

    def execute_epoch(self):
        raise NotImplementedError()
    # endregion
