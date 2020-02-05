from typing import Union, Dict, Any, List

import numpy as np
import pandas as pd

from domain.Enums import Status, HttpCodes
from domain.Hive import Hive
from domain.SharedFilePart import SharedFilePart
from globals.globals import DEFAULT_COLUMN, REPLICATION_LEVEL
from utils import crypto
from utils.ResourceTracker import ResourceTracker as rT


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
    def init_recovery_protocol(self, file_name: str) -> SharedFilePart:
        """
        Reconstructs a file and then splits it into globals.READ_SIZE before redistributing them to the rest of the hive
        :param str file_name: id of the shared file that needs to be reconstructed by the Worker instance
        """
        # TODO future-iterations:
        raise NotImplementedError()
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
    def send_part(self, hive: Hive, part: SharedFilePart) -> Union[int, HttpCodes]:
        """
        Attempts to send a file part to another worker
        :param Hive hive: Gateway hive that will deliver this file to other worker
        :param SharedFilePart part: data class instance with data w.r.t. the shared file part and it's raw contents
        """
        routing_vector: pd.DataFrame = self.routing_table[part.name]
        hive_members: List[str] = [*routing_vector.index]
        member_chances: List[float] = [*routing_vector.iloc[:, DEFAULT_COLUMN]]
        destination: str = np.random.choice(a=hive_members, p=member_chances).item()  # converts numpy.str to python str
        if destination == self.id:
            return HttpCodes.DUMMY
        return hive.route_part(destination, part)

    def reroute_part(self, hive: Hive, part: SharedFilePart) -> None:
        """
        Tries to send a part until it delivers it to someone else
        :param Hive hive: Hive instance that delivered the part on behalf of another worker
        :param SharedFilePart part: file part that needs to be rerouted to avoid duplicates
        """
        response_code = HttpCodes.DUMMY
        while response_code != HttpCodes.OK:
            response_code = self.send_part(hive, part)

    def receive_part(self, part: SharedFilePart) -> int:
        """
        Keeps a new, single, shared file part, along the ones already stored by the Worker instance
        :param SharedFilePart part: data class instance with data w.r.t. the shared file part and it's raw contents
        :returns HttpCodes int
        """
        if part.name not in self.files:
            self.files[part.name] = {}  # init dict that accepts <key: id, value: sfp> pairs for the file

        if crypto.sha256(part.data) != part.sha256:
            return HttpCodes.BAD_REQUEST  # inform sender that his part is corrupt, don't initiate recovery protocol, to avoid denial of service attacks on worker
        elif part.number in self.files[part.name]:
            return HttpCodes.NOT_ACCEPTABLE  # reject repeated replicas even if they are correct
        else:
            self.files[part.name][part.number] = part
            return HttpCodes.OK  # accepted file part, because Sha256 was correct and Worker did not have this replica yet
    # endregion

    # region Swarm Guidance Interface
    def execute_epoch(self, hive: Hive, file_name: str) -> None:
        """
        For each part kept by the Worker instance, get the destination and send the part to it
        :param Hive hive: Hive instance that ordered execution of the epoch
        :param str file_name: the file parts that should be routed
        """
        file_cache: Dict[int, SharedFilePart] = self.files.get(file_name, {})
        epoch_cache: Dict[int, SharedFilePart] = {}
        for number, part in file_cache.items():
            replicate: int = part.can_replicate()  # Number of times that file part needs to be replicated to achieve REPLICATION_LEVEL
            if replicate:
                i: int = 0
                part.reset_epochs_to_recover()  # Ensures other workers don't try to replicate and that Hive can resimulate delays
                while i < replicate and i < hive.hive_size:  # Second condition avoids infinite loop where hive_size < REPLICATION_LEVEL and Workers keep rerouting to each other
                    i += 1
                    part.references += 1
                    self.reroute_part(hive, part)
            response_code = self.send_part(hive, part)
            if response_code == HttpCodes.BAD_REQUEST:
                part = self.init_recovery_protocol(part.name)  # TODO: future-iterations, right now, nothing is returned by init recovery protocol
                epoch_cache[number] = part
            elif response_code != HttpCodes.OK:
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
