import uuid
import numpy as np
import pandas as pd
import utils.matrices as matrices
import utils.metropolis_hastings as mh

from domain.Worker import Worker
from domain.Enums import Status, HttpCodes
from domain.helpers.FileData import FileData
from domain.SharedFilePart import SharedFilePart

from typing import Dict, List, Union, Any


class Hive:
    """
    :ivar str hive_id: unique identifier in str format
    :ivar Dict[str, Worker] members: Workers that belong to this P2P Hive, key is worker.id, value is the respective Worker instance
    :ivar Dict[str, FileData] files: maps the name of the files persisted by the members of this Hive, to instances of FileData used by the Simulator class
    :ivar DataFrame desired_distribution: distribution hive members are seeking to achieve for each the files they persist together.
    """

    # region Class Variables, Instance Variables and Constructors
    def __init__(self) -> None:
        """
        Instantiates an Hive abstraction
        """
        self.hive_id: str = str(uuid.uuid4())
        self.members: Dict[str, Worker] = {}
        self.files: Dict[str, FileData] = {}
        self.desired_distribution: pd.DataFrame = pd.DataFrame()
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
        desired_distribution = self.new_desired_distribution(member_uptimes)

        transition_matrix: np.ndarray = mh.metropolis_algorithm(adjancency_matrix, desired_distribution, column_major_out=True)
        return pd.DataFrame(transition_matrix, index=member_ids, columns=member_ids)

    def broadcast_transition_matrix(self) -> None:
        """
        Gives each member his respective slice (vector column) of the transition matrix the Hive is currently executing.
        post-scriptum: we could make an optimization that sets a transition matrix for the hive, ignoring the file names, instead of mapping different file
        names to an equal transition matrix within each hive member, thus reducing space overhead arbitrarly, however, this would make Simulation harder. This
        note is kept for future reference.
        """
        transition_vector: pd.DataFrame
        transition_matrix: pd.DataFrame = self.new_transition_matrix()

        for worker in self.members.values():
            transition_vector = transition_matrix.loc[:, worker.id]
            for file in self.files.values():
                worker.set_file_routing(file.name, transition_vector)
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


