import uuid

from typing import Set, Union
from domain.Worker import Worker


class Hive:
    """
    :ivar Set[Worker] members: Workers that belong to this P2P Hive
    :ivar str hive_id: unique identifier in str format
    :ivar str file_name:
    """
    def __init__(self, file_name: str) -> None:
        """
        Instantiates an Hive abstraction
        :param str file_name: name of the file that is managed under this hive
        """
        self.members: Set[Worker] = set()
        self.hive_id: str = str(uuid.uuid4())
        self.file_name = file_name

    def set_transition_matrix(self, transition_matrix) -> None:
        """
        Gives each member his respective vector column belonging to the transition matrix the Hive is now executing.
        """
        # TODO: transition matrix should be a vector
        for worker in self.members:
            worker.set_file_routing(self.file_name, transition_matrix)

    def remove_member(self, worker: Union[str, Worker]) -> None:
        """
        Removes a worker from the Hive's membership set.
        :param Union[str, Worker] worker: name of the worker or instance object of class Worker to be removed from the set.
        """
        self.members.discard(worker)

    def add_member(self, worker: Worker) -> None:
        """
        Adds a worker to the Hive's membership set.
        :param Worker worker: instance object of class Worker to be added to the set.
        """
        self.members.add(worker)

    def replace_member(self, old: Union[str, Worker], new: Worker) -> None:
        """
        Replaces a worker from the Hive's membership set with some other worker.
        :param Union[str, Worker] old: name of the worker or instance object of class Worker to be replaced in the set.
        :param Worker new: instance object of class Worker to be added to the set.
        """
        self.members.discard(old)
        self.members.add(new)
