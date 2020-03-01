from __future__ import annotations

import math
import sys
import traceback
import numpy as np
import pandas as pd
import domain.Hive as h

from utils import crypto
from typing import Union, Dict, Any, List
from domain.helpers.Enums import Status, HttpCodes
from domain.helpers.SharedFilePart import SharedFilePart
from globals.globals import DEFAULT_COL, MAX_EPOCHS
from utils.ResourceTracker import ResourceTracker as rT


class Worker:
    """
    Defines a worker node on the P2P network.
    :cvar List[Union[Status.ONLINE, Status.OFFLINE, Status.SUSPECT]] ON_OFF: possible Worker states, will be extended to include suspect down the line
    :ivar str id: unique identifier of the worker instance on the network
    :ivar float uptime: average worker uptime
    :ivar Dict[str, Hive] hives: dictionary that maps the hive_ids' this worker instance belongs to, to the respective Hive instances
    :ivar Dict[str, Dict[int, SharedFilePart]] files: collection mapping file names to file parts and their contents
    :ivar Dict[str, pd.DataFrame] routing_table: collection mapping file names to the respective transition probabilities followed by the worker instance
    :ivar Union[int, Status] status: indicates if this worker instance is online or offline, might have other non-intuitive status, hence bool does not suffice
    """

    # region Class Variables, Instance Variables and Constructors
    def __init__(self, worker_id: str, worker_uptime: float):
        self.id: str = worker_id
        self.uptime: float = float('inf') if worker_uptime == 1.0 else math.ceil(worker_uptime * MAX_EPOCHS)
        self.hives: Dict[str, h.Hive] = {}
        self.files: Dict[str, Dict[int, SharedFilePart]] = {}
        self.routing_table: Dict[str, pd.DataFrame] = {}
        self.status: int = Status.ONLINE
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
    def send_part(self, hive: h.Hive, part: SharedFilePart) -> Union[int, HttpCodes]:
        """
        Attempts to send a file part to another worker
        :param Hive hive: Gateway hive that will deliver this file to other worker
        :param SharedFilePart part: data class instance with data w.r.t. the shared file part and it's raw contents
        :returns HttpCodes: response obtained from sending the part
        """
        routing_vector: pd.DataFrame = self.routing_table[part.name]
        hive_members: List[str] = [*routing_vector.index]
        member_chances: List[float] = [*routing_vector.iloc[:, DEFAULT_COL]]
        try:
            destination: str = np.random.choice(a=hive_members, p=member_chances).item()  # converts numpy.str to python str
            return hive.route_part(self.id, destination, part)
        except ValueError as vE:
            print(routing_vector)
            sys.exit("".join(traceback.format_exception(etype=type(vE), value=vE, tb=vE.__traceback__)))

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

    def replicate(self, hive: h.Hive, part: SharedFilePart) -> None:
        """
        Equal to send part but with different semantics, as file is not routed following swarm guidance, but instead by choosing the most reliable peers in the hive
        post-scriptum: This function is hacked... And should only be used for simulation purposes
        :param Hive hive: Gateway hive that will deliver this file to other worker
        :param SharedFilePart part: data class instance with data w.r.t. the shared file part and it's raw contents
        """
        lost_replicas: int = part.can_replicate(hive.current_epoch)  # Number of times that file part needs to be replicated to achieve REPLICATION_LEVEL
        if lost_replicas > 0:
            sorted_member_view: List[str] = [*hive.desired_distribution.sort_values(DEFAULT_COL, ascending=False).index]
            for member_id in sorted_member_view:
                if lost_replicas == 0:
                    break
                elif hive.route_part(self.id, member_id, part, fresh_replica=True) == HttpCodes.OK:
                    lost_replicas -= 1
                    part.references += 1
            part.reset_epochs_to_recover(hive.current_epoch)
    # endregion

    # region Swarm Guidance Interface
    def execute_epoch(self, hive: h.Hive, file_name: str) -> None:
        """
        For each part kept by the Worker instance, get the destination and send the part to it
        :param Hive hive: Hive instance that ordered execution of the epoch
        :param str file_name: the file parts that should be routed
        """
        file_view: Dict[int, SharedFilePart] = self.files.get(file_name, {}).copy()
        for number, part in file_view.items():
            self.replicate(hive, part)
            response_code = self.send_part(hive, part)
            if response_code == HttpCodes.OK:
                self.discard_part(file_name, number)
            elif response_code == HttpCodes.BAD_REQUEST:
                self.discard_part(file_name, number, corrupt=True, hive=hive)
            elif HttpCodes.TIME_OUT or HttpCodes.NOT_ACCEPTABLE or HttpCodes.DUMMY:
                pass  # Keep file part for at least one more epoch
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
    def discard_part(self, name: str, number: int, corrupt: bool = False, hive: h.Hive = None) -> None:
        """
        # TODO future-iterations: refactor to work with multiple file names
        Safely deletes a part from the worker instance's cache
        :param str name: name of the file the part belongs to
        :param int number: the part number that uniquely identifies it
        :param bool corrupt: if discard is due to corruption
        :param Hive hive:
        """
        part: SharedFilePart = self.files.get(name, {}).pop(number, None)
        if part and corrupt:
            if part.decrease_and_get_references() == 0:
                hive.set_fail("lost all replicas of file part with id: {}, and last loss was due to corruption".format(part.id))
            else:
                hive.set_recovery_epoch(part)

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

    def get_epoch_status(self) -> int:
        """
        When called, the worker instance decides if it should switch status
        """
        if self.status == Status.ONLINE:
            self.uptime -= 1
            self.status = Status.ONLINE if self.uptime > 0 else Status.OFFLINE
        return self.status

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
