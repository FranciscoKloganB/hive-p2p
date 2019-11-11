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
    # region docstrings
    """
    Defines a node on the P2P network. Workers are subject to constraints imposed by Hivemind, constraints they inflict
    on themselves based on available computing power (CPU, RAM, etc...) and can have [0, N] shared file parts. Workers
    have the ability to reconstruct lost file parts when needed.
    :ivar sf_parts: key part_name maps to a dict of part_id keys whose values are SharedFilePart
    :type dict<str, dict<str, SharedFilePart>
    :ivar name: id of this worker node that uniquely identifies him in the network
    :type str
    :ivar hivemind: coordinator of the unstructured Hybrid P2P network that enlisted this worker for a Hive
    :type str
    :ivar routing_table: maps file name with state transition probabilities, from this worker to other workers
    :type dict<str, pandas.DataFrame>
    """
    # endregion

    # region class variables, instance variables and constructors
    def __init__(self, hivemind, name):
        self.sf_parts = {}
        self.routing_table = {}
        self.name = name
        self.hivemind = hivemind
    # endregion

    # region overriden class methods
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
        :param str sf_name: name of the shared file that needs to be reconstructed by the Worker instance
        :type str
        """
        # TODO future-iterations:
        #  1. Recovery algorithm
        raise NotImplementedError
    # endregion

    # region routing table management methods
    def set_file_routing(self, sf_name: str, transition_vector: Union[pd.Series, pd.DataFrame]) -> None:
        """
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
            self.sf_parts.pop(sf_name)
        except KeyError as kE:
            log.error("Key ({}) doesn't exist in worker {}'s sf_parts dict".format(sf_name, self.name))
            log.error("Key Error message: {}".format(str(kE)))
    # endregion

    # region file sending and receiving methods
    def receive_part(self, part: SharedFilePart, no_check: bool = False) -> None:
        """
        Keeps a new, single, shared file part, along the ones already stored by the Worker instance
        :param SharedFilePart part: data class instance with data w.r.t. the shared file part and it's raw contents
        :param bool no_check: wether or not method verifies sha256 of the received part
        """
        if no_check or crypto.sha256(part.part_data) == part.sha256:
            if part.part_name not in self.sf_parts:
                self.sf_parts[part.part_name] = {}  # init dict that accepts <key: id, value: sfp> pairs for the file
            self.sf_parts[part.part_name][part.part_id] = part
        else:
            print("part_name: {}, part_number: {} - corrupted".format(part.part_name, str(part.part_number)))
            self.init_recovery_protocol(part.part_name)

    def receive_parts(self, sf_id_sfp_dict: Dict[int, SharedFilePart], sf_name: str = None, no_check: bool = False) -> None:
        """
        Keeps incomming shared file parts along with the ones already owned by the Worker instance
        :param Dict[int, SharedFilePart] sf_id_sfp_dict: mapping of shared file part id to SharedFileParts instances
        :param str sf_name: name of the file the parts belong to. If no_check is True, than, sf_name must be set!
        :param bool no_check: wether or not method verifies sha256 of each part.
        """
        if no_check and sf_name is not None:
            self.sf_parts[sf_name].update(sf_id_sfp_dict)  # Appends sf_id_sfp_dict values to existing values
        else:
            for sf_part in sf_id_sfp_dict.values():  # receive_part(...) automatically fetches the part_number for part
                self.receive_part(sf_part, no_check=False)

    def route_parts(self) -> None:
        """
        For each part kept by the Worker instance, get the destination and send the part to it
        """
        sf_parts_dict = self.sf_parts.items()

        for sf_name, sf_id_sfp_dict in sf_parts_dict:

            tmp = {}
            sf_id_sfp_dict = sf_id_sfp_dict.items()

            for sf_id, sf_part in sf_id_sfp_dict:
                dest_worker = self.get_next_state(sf_name=sf_name)
                if dest_worker == self.name:
                    tmp[sf_id] = sf_part  # store <sf_id, sf_part> pair in tmp dict, we don't need to send to ourselves
                else:
                    response_code = self.hivemind.route_file_part(dest_worker, sf_part)
                    if response_code != HttpCodes.OK:
                        self.hivemind.receive_complaint(dest_worker, sf_name=sf_name)
                        tmp[sf_id] = sf_part  # store <sf_id, sf_part>, original destination doesn't respond

            self.sf_parts[sf_name] = tmp  # update sf_parts[sf_name] with all parts that weren't transmited
    # endregion

    # region helpers
    def get_parts_count(self, sf_name: str) -> int:
        """
        :param sf_name: name of the file kept by the Worker instance that must be counted
        :returns int: number of parts from the named shared file currently on the Worker instance
        """
        return len(self.sf_parts[sf_name])

    def get_next_state(self, sf_name: str) -> str:
        """
        :param str sf_name: the name of the file the part to be routed belongs to
        :returns str: the name of the worker to whom the file should be routed too
        """
        routing_data = self.routing_table[sf_name]
        row_labels = [*routing_data.index]  # gets the names of sharers as a list
        # label_probabilities = [*routing_data.iloc[:, DEFAULT_COLUMN]]  # probabilities corresponding to labeled sharer
        label_probabilities = [*routing_data.iloc[:, DEFAULT_COLUMN]]  # probabilities corresponding to labeled sharer
        return np.random.choice(a=row_labels, p=label_probabilities).item()  # converts numpy.str to python str
    # endregion

    # region resource utilization methods
    @staticmethod
    def get_resource_utilization(*args) -> Dict[str, Any]:
        """
        :param *args: Variable length argument list. See below
        :keyword arg:
        :arg 'cpu': system wide float detailing cpu usage as a percentage,
        :arg 'cpu_count': number of non-logical cpu on the machine as an int
        :arg 'cpu_avg': average system load over the last 1, 5 and 15 minutes as a tuple
        :arg 'mem': statistics about memory usage as a named tuple including the following fields (total, available),
        expressed in bytes as floats
        :arg 'disk': get_disk_usage dictionary with total and used keys (gigabytes as float) and percent key as float
        :returns Dict[str, Any] detailing the usage of the respective key arg. If arg is invalid the value will be -1.
        """
        results = {}
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
        return deepcopy(self.sf_parts)
    # endregion
