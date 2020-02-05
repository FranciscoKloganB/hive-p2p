import numpy as np
import logging as log
import pandas as pd

from utils import crypto
from copy import deepcopy
from globals.globals import DEFAULT_COLUMN
from domain.Hive import Hive
from domain.Enums import Status, HttpCodes
from domain.SharedFilePart import SharedFilePart
from utils.ResourceTracker import ResourceTracker as rT

from typing import Union, Dict, Any, List


class Worker:
    """
    Defines a worker node on the P2P network.
    :cvar List[Union[Status.ONLINE, Status.OFFLINE, Status.SUSPECT]] ON_OFF: possible Worker states, will be extended to include suspect down the line
    :ivar str id: unique identifier of the worker instance on the network
    :ivar float uptime: average worker uptime
    :ivar float disconnect_chance: (100.0 - worker_uptime) / 100.0
    :ivar Dict[str, Hive] hives: dictionary that maps the hive_ids' this worker instance belongs to, to the respective Hive instances
    :ivar Dict[str, Dict[int, SharedFilePart]] files: collection mapping file names to file parts and their contents
    :ivar Dict[str, pd.DataFrame] routing_table: collection mapping file names to the respective transition probabilities followed by the worker instance
    :ivar Union[int, Status] status: indicates if this worker instance is online or offline, might have other non-intuitive status, hence bool does not suffice
    """

    ON_OFF: List[Union[Status.ONLINE, Status.OFFLINE, Status.SUSPECT]] = [Status.ONLINE, Status.SUSPECT]

    # region Class Variables, Instance Variables and Constructors
    def __init__(self, worker_id: str, worker_uptime: float):
        self.id: str = worker_id
        self.uptime: float = worker_uptime
        self.disconnect_chance: float = 1.0 - worker_uptime
        self.hives: Dict[str, Hive] = {}
        self.files: Dict[str, Dict[int, SharedFilePart]] = {}
        self.routing_table: Dict[str, pd.DataFrame] = {}
        self.status: Union[int, Status] = Status.ONLINE
    # endregion

    # region Recovery
    def init_recovery_protocol(self, file_name: str) -> None:
        """
        Reconstructs a file and then splits it into globals.READ_SIZE before redistributing them to the rest of the hive
        :param str file_name: id of the shared file that needs to be reconstructed by the Worker instance
        """
        # TODO future-iterations:
        pass
    # endregion

    # region Routing Table
    def set_file_routing(self, file_name: str, transition_vector: Union[pd.Series, pd.DataFrame]) -> None:
        """
        Maps file id with state transition probabilities used for routing
        :param str file_name: a file id that is being shared on the hive
        :param Union[pd.Series, pd.DataFrame] transition_vector: probabilities of going from current worker to some worker on the hive
        """
        if isinstance(transition_vector, pd.Series):
            self.routing_table[file_name] = transition_vector.to_frame()
        elif isinstance(transition_vector, pd.DataFrame):
            self.routing_table[file_name] = transition_vector
        else:
            raise ValueError("Worker.set_file_routing expects a pandas.Series or pandas.DataFrame as type for transition vector parameter.")

    def remove_file_routing(self, file_name: str) -> None:
        """
        Removes a shared file's routing information from the routing table
        :param str file_name: id of the shared file whose routing information is being removed from routing_table
        """
        self.files.pop(file_name)
    # endregion

    # region File Routing
    def send_part(self, part: SharedFilePart) -> Union[int, HttpCodes]:
        """
        Attempts to send a file part to another worker
        :param SharedFilePart part: data class instance with data w.r.t. the shared file part and it's raw contents
        """
        routing_vector: pd.DataFrame = self.routing_table[part.name]
        hive_members: List[str] = [*routing_vector.index]
        member_chances: List[float] = [*routing_vector.iloc[:, DEFAULT_COLUMN]]
        destination: str = np.random.choice(a=hive_members, p=member_chances).item()  # converts numpy.str to python str
        if destination == self.id:
            return HttpCodes.DUMMY
        return self.hives[part.hive_id].route_part(destination, part)

    def reroute_part(self, part: SharedFilePart) -> None:
        response_code = HttpCodes.DUMMY
        while response_code != HttpCodes.OK:
            response_code = self.send_part(part)

    def receive_part(self, part: SharedFilePart) -> None:
        """
        Keeps a new, single, shared file part, along the ones already stored by the Worker instance
        :param SharedFilePart part: data class instance with data w.r.t. the shared file part and it's raw contents
        """
        if part.name not in self.files:
            self.files[part.name] = {}  # init dict that accepts <key: id, value: sfp> pairs for the file

        if part.number in self.files[part.name]:
            self.reroute_part(part)  # pass part's replica to someone else, who might or might not have it.
        elif crypto.sha256(part.data) == part.sha256:
            self.files[part.name][part.number] = part  # if sha256 is correct and worker does not have a replica, he keeps it
        else:
            print("shared file part id: {}, corrupted".format(part.id))
            self.init_recovery_protocol(part.name)
    # endregion

    # region Swarm Guidance Interface
    def execute_epoch(self, file_name: str) -> None:
        """
        For each part kept by the Worker instance, get the destination and send the part to it
        :param str file_name: the file parts that should be routed
        """
        hive: Hive
        file_cache: Dict[int, SharedFilePart] = self.files.get(file_name, {})
        epoch_cache: Dict[int, SharedFilePart] = {}
        for number, part in file_cache.items():
            response_code = self.send_part(part)
            if response_code != HttpCodes.OK:
                epoch_cache[number] = part
        self.files[file_name] = epoch_cache
    # endregion

    # region PSUtils Interface
    # noinspection PyIncorrectDocstring
    @staticmethod
    def get_resource_utilization(*args) -> Dict[str, Any]:
        """
        Obtains one ore more performance attributes for the Worker's instance machine
        :param *args: Variable length argument list. See below
        :keyword arg:
        :arg cpu: system wide float detailing cpu usage as a percentage,
        :arg cpu_count: number of non-logical cpu on the machine as an int
        :arg cpu_avg: average system load over the last 1, 5 and 15 minutes as a tuple
        :arg mem: statistics about memory usage as a named tuple including the following fields (total, available),
        expressed in bytes as floats
        :arg disk: get_disk_usage dictionary with total and used keys (gigabytes as float) and percent key as float
        :returns Dict[str, Any] detailing the usage of the respective key arg. If arg is invalid the value will be -1.
        """
        results: Dict[str, Any] = {}
        for arg in args:
            results[arg] = rT.get_value(arg)
        return results
    # endregion

    # region Overrides
    def __hash__(self):
        # allows a worker object to be used as a dictionary key
        return hash(str(self.id))

    def __eq__(self, other):
        return self.id == other

    def __ne__(self, other):
        return not(self == other)
    # endregion

    # region Helpers
    def get_file_parts(self, file_name: str) -> Dict[int, SharedFilePart]:
        """
        Gets collection of file parts that correspond to the named file
        :param str file_name: the file parts that caller wants to retrieve from this worker instance
        :returns Dict[int, SharedFilePart]: reference to a collection that maps part numbers to file parts
        """
        return self.files.get(file_name, {})

    def get_file_parts_count(self, file_name: str) -> int:
        """
        Counts how many parts the Worker instance has of the named file
        :param str file_name: the file parts that caller wants to count
        :returns int: number of parts from the named shared file currently on the Worker instance
        """
        return len(self.files.get(file_name, {}))

    def get_epoch_status(self) -> Union[Status.ONLINE, Status.OFFLINE, Status.SUSPECT]:
        """
        When called, the worker instance decides if it should switch status
        """
        if self.status == Status.OFFLINE:
            return self.status
        else:
            self.status = np.random.choice(Worker.ON_OFF, p=[self.uptime, self.disconnect_chance])
            return self.status
    # endregion
