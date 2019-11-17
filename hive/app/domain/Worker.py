import numpy as np
import logging as log
import pandas as pd

from utils import crypto
from copy import deepcopy
from domain.Enums import HttpCodes
from typing import Dict, Union, Any
from globals.globals import DEFAULT_COLUMN
from domain.SharedFilePart import SharedFilePart
from utils.ResourceTracker import ResourceTracker as rT


class Worker:
    """
    Defines a worker node on the P2P network.
    :ivar Dict[str, Dict[int, SharedFilePart]] shared_files: collection of file parts kept by the worker instance
    :ivar str name: unique identifier of the worker instance on the network
    :ivar Hivemind hivemind: super node managing the Worker instance
    :ivar Dict[str, pd.DataFrame] routing_table: maps file names with their state transition probabilities
    """

    # region class variables, instance variables and constructors
    def __init__(self, hivemind: Any, name: str):
        self.hivemind: Any = hivemind
        self.name: str = name
        self.shared_files: Dict[str, Dict[int, SharedFilePart]] = {}
        self.routing_table: Dict[str, pd.DataFrame] = {}
    # endregion

    # region override class methods
    def __hash__(self):
        # allows a worker object to be used as a dictionary key
        return hash(str(self.name))

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        return (self.hivemind, self.name) == (other.hivemind, other.name)

    def __ne__(self, other):
        return not(self == other)
    # endregion

    # region file recovery methods
    def init_recovery_protocol(self, sf_name: str) -> None:
        """
        Reconstructs a file and then splits it into globals.READ_SIZE before redistributing them to the rest of the hive
        :param str sf_name: name of the shared file that needs to be reconstructed by the Worker instance
        """
        # TODO future-iterations:
        #  1. Recovery algorithm
        raise NotImplementedError
    # endregion

    # region routing table management methods
    def set_file_routing(self, sf_name: str, transition_vector: Union[pd.Series, pd.DataFrame]) -> None:
        """
        Maps file name with state transition probabilities
        :param str sf_name: a file name that is being shared on the hive
        :param Union[pd.Series, pd.DataFrame] transition_vector: probabilities of going from current worker to some
         worker on the hive
        """
        if isinstance(transition_vector, pd.Series):
            self.routing_table[sf_name] = transition_vector.to_frame()
        elif isinstance(transition_vector, pd.DataFrame):
            self.routing_table[sf_name] = transition_vector
        else:
            raise ValueError("Worker.set_file_routing expects a pandas.Series or pandas.DataFrame transition vector.")

    def update_file_routing(self, sf_name: str, replacement_dict: Dict[str, str]) -> None:
        """
        Updates a shared file's routing information within the routing table
        :param str sf_name: name of the shared file whose routing information is being updated
        :param Dict[str, str] replacement_dict: old worker name, new worker name)
        """
        self.routing_table[sf_name].rename(index=replacement_dict, inplace=True)

    def remove_file_routing(self, sf_name: str) -> None:
        """
        Removes a shared file's routing information from within the routing table
        :param str sf_name: name of the shared file whose routing information is being removed from routing_table
        """
        try:
            self.shared_files.pop(sf_name)
        except KeyError as kE:
            log.error("Key ({}) doesn't exist in worker {}'s sf_parts dict".format(sf_name, self.name))
            log.error("Key Error message: {}".format(str(kE)))
    # endregion

    # region file sending and receiving methods
    def receive_part(self, sfp: SharedFilePart, no_check: bool = False) -> None:
        """
        Keeps a new, single, shared file part, along the ones already stored by the Worker instance
        :param SharedFilePart sfp: data class instance with data w.r.t. the shared file part and it's raw contents
        :param bool no_check: whether or not method verifies sha256 of the received part
        """
        if no_check or crypto.sha256(sfp.part_data) == sfp.sha256:
            if sfp.part_name not in self.shared_files:
                self.shared_files[sfp.part_name] = {}  # init dict that accepts <key: id, value: sfp> pairs for the file
            self.shared_files[sfp.part_name][sfp.part_id] = sfp
        else:
            print("part_name: {}, part_number: {} - corrupted".format(sfp.part_name, str(sfp.part_number)))
            self.init_recovery_protocol(sfp.part_name)

    def receive_parts(self, sf_id_sfp_dict: Dict[int, SharedFilePart], sf_name: str = None, no_check: bool = False) -> None:
        """
        Keeps incoming shared file parts along with the ones already owned by the Worker instance
        :param dict sf_id_sfp_dict: mapping of shared file part id to SharedFileParts instances
        :param str sf_name: name of the file the parts belong to.
        :param bool no_check: whether or not method verifies sha256 of each part.
        """
        if sf_name:
            self.__update_shared_files_dict(sf_id_sfp_dict, sf_name, no_check)
        else:
            for sf_part in sf_id_sfp_dict.values():  # receive_part(...) automatically fetches the part_number for part
                self.receive_part(sf_part, no_check)

    def route_parts(self) -> None:
        """
        For each part kept by the Worker instance, get the destination and send the part to it
        """
        sf_parts_dict = self.shared_files.items()

        for sf_name, sf_id_sfp_dict in sf_parts_dict:

            tmp: Dict[int, SharedFilePart] = {}
            sf_id_sfp_dict = sf_id_sfp_dict.items()

            for sf_id, sf_part in sf_id_sfp_dict:
                dest_worker = self.get_next_state(sf_name=sf_name)
                if dest_worker == self.name:
                    tmp[sf_id] = sf_part  # store <sf_id, sf_part> pair in tmp dict, we don't need to send to ourselves
                else:
                    response_code = self.hivemind.route_file_part(dest_worker, sf_part)
                    if response_code != HttpCodes.OK:
                        self.hivemind.receive_complaint(dest_worker)
                        tmp[sf_id] = sf_part  # store <sf_id, sf_part>, original destination doesn't respond

            self.shared_files[sf_name] = tmp  # update sf_parts[sf_name] with all parts that weren't transmited
    # endregion

    # region helpers
    def get_parts_count(self, sf_name: str) -> int:
        """
        Counts how many parts the Worker instance has of the named shared file
        :param sf_name: name of the file kept by the Worker instance that must be counted
        :returns int: number of parts from the named shared file currently on the Worker instance
        """
        return len(self.shared_files[sf_name])

    def get_next_state(self, sf_name: str) -> str:
        """
        Selects the next destination for a shared file part
        :param str sf_name: the name of the file the part to be routed belongs to
        :returns str: the name of the worker to whom the file should be routed too
        """
        routing_data: pd.DataFrame = self.routing_table[sf_name]
        row_labels: List[str] = [*routing_data.index]
        label_probabilities: List[float] = [*routing_data.iloc[:, DEFAULT_COLUMN]]
        return np.random.choice(a=row_labels, p=label_probabilities).item()  # converts numpy.str to python str
    # endregion

    # region resource utilization methods
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

    # region mock methods
    def get_all_parts(self) -> Dict[str, Dict[int, SharedFilePart]]:
        """
        Sends all shared file parts kept by the Worker instance to the requestor regardless of the file's hive
        :returns Dict[str, Dict[int, SharedFilePart]]: a deep copy of the Worker's instance shared file parts
        """
        return deepcopy(self.shared_files)
    # endregion

    # region helpers
    def __update_shared_files_dict(
            self, sf_id_sfp_dict: Dict[int, SharedFilePart], sf_name: str = None, no_check: bool = False) -> None:
        """
        Creates a key value pair in the Worker instance shared_files field or updates the existing one with more parts
        :param sf_id_sfp_dict: collection mapping part.id to SharedFilePart instances
        :param sf_name: the named of the shared file
        :param no_check: if integrity of the shared file parts in the dictionary need to be checked or not
        """
        if (not no_check) and (self.__sf_id_sfp_dict_needs_fix(sf_id_sfp_dict)):
            # if check == True and if any part as an incorrect sha256, then, fix file and return
            self.init_recovery_protocol(sf_name)
        else:
            # if (check == False) OR (check == True and all parts have correct sha256), then, update dict accordingly
            if sf_name in self.shared_files:
                self.shared_files[sf_name].update(sf_id_sfp_dict)  # Appends sf_id_sfp_dict values to existing values
            else:
                self.shared_files[sf_name] = sf_id_sfp_dict

    def __sf_id_sfp_dict_needs_fix(self, sf_id_sfp_dict: Dict[int, SharedFilePart]) -> bool:
        """
        Verifies the integrity of all shared file parts in the inputed dictionary
        :param sf_id_sfp_dict: collection mapping part.id to SharedFilePart instances
        :returns bool: True if any of the parts in the dictionary fails the integrity test, False if all are deemed OK
        """
        for sfp in sf_id_sfp_dict.values():
            if crypto.sha256(sfp.part_data) != sfp.sha256:
                return True
        return False

    # endregion
