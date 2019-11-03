import numpy as np
import logging as log
from utils import crypto
from copy import deepcopy
from utils.ResourceTracker import ResourceTracker as rT
from domain.Enums import HttpCodes


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
        self.__routing_table = {}
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
    def init_recovery_protocol(self, sf_name):
        """
        :param sf_name: name of the file that needs to be reconstructed and redistributed
        :type str
        """
        # TODO future-iterations:
        #  1. Recovery algorithm
        raise NotImplementedError
    # endregion

    # region routing table management methods
    def set_file_routing(self, sf_name, transition_vector):
        """
        :param sf_name: a file name that is being shared on the hive
        :type str
        :param transition_vector: probability vector indicating transitions to other states for the given file w/ labels
        :type 1-D numpy.Array in column format
        """
        self.__routing_table[sf_name] = transition_vector

    def update_file_routing(self, sf_name, replacement_dict):
        """
        :param sf_name: a file name that is being shared on the hive
        :type str
        :param replacement_dict: key, value pair where key represents the name to be replaced with the new value
        :type dict<str, str>
        """
        self.__routing_table[sf_name].rename(index=replacement_dict, inplace=True)

    def remove_file_routing(self, sf_name):
        """
        :param sf_name: a file name that is being shared on the hive which this node should stop transimitting
        :type str
        """
        try:
            self.sf_parts.pop(sf_name)
        except KeyError:
            log.error("Key ({}) doesn't exist in worker {}'s sf_parts dict".format(sf_name, self.name))
    # endregion

    # region file sending and receiving methods
    def receive_part(self, part, no_check=False):
        """
        Keeps a new, single, shared file part, along the ones already stored by the Worker instance
        :param part: an instance object that contains data regarding the shared file part and it's raw contents
        :type domain.SharedFilePart
        :param no_check: wether or not method verifies sha256 of each part.
        :type bool
        """
        if no_check or crypto.sha256(part.part_data) == part.sha256:
            if part.name in self.sf_parts:
                self.sf_parts[part.name][part.part_id] = part
            else:
                self.sf_parts[part.name] = {}
                self.sf_parts[part.name][part.part_id] = part
        else:
            print("part_name: {}, part_number: {} - corrupted".format(part.part_name, str(part.part_number)))
            self.init_recovery_protocol(part)

    def receive_parts(self, sf_id_parts, sf_name=None, no_check=False):
        """
        Keeps incomming shared file parts along with the ones already owned by the Worker instance
        :param sf_id_parts: mapping of shared file part id to SharedFileParts instances
        :type dict<int, domain.SharedFilePart>
        :param sf_name: name of the file the parts belong to. sf_name must be set if method is called with no_check=True
        :type str
        :param no_check: wether or not method verifies sha256 of each part.
        :type bool
        """
        if no_check and sf_name is not None:
            # Use sf_name in param to leverage the pythonic way of merging dictionaries
            self.sf_parts[sf_name].update(sf_id_parts)
        else:
            for sf_part in sf_id_parts.values():
                # When adding one 1-by-1, no other param other than the SharedFilePart is needed... See receive_part
                self.receive_part(sf_part, no_check=False)

    def route_parts(self):
        """
        For each part kept by the Worker instance, get the destination and send the part to it
        """
        for sf_name, sf_id in self.sf_parts.items():
            tmp = {}
            for sf_part in sf_id.values():
                dest_worker = self.get_next_state(shared_file_name=sf_name)
                if dest_worker == self.name:
                    tmp[sf_id] = sf_part  # store <sf_id, sf_part> pair in tmp dict, we don't need to send to ourselves
                else:
                    response_code = self.hivemind.route_file_part(dest_worker, sf_part)
                    if response_code != HttpCodes.OK:
                        self.hivemind.receive_complaint(dest_worker, sf_name=sf_name)
                        tmp[sf_id] = sf_part  # store <sf_id, sf_part>, original destination doesn't respond
            self.sf_parts[sf_name] = tmp  # update sf_parts[sf_name] with all parts that weren't transmited

    # region helpers
    def leave_hive(self):
        """
        Worker instance disconnects from the Hivemind and discards all files it is currently keeping
        """
        self.hivemind = None
        self.sf_parts = None

    def get_next_state(self, shared_file_name):
        """
        :param shared_file_name: the name of the file the part to be routed belongs to
        :type: str
        :return: the name of the worker to whom the file should be routed too
        :type: str
        """
        routing_data = self.__routing_table[shared_file_name]
        row_labels = [*routing_data.index]  # gets the names of sharers as a list
        label_probabilities = [*routing_data[self.name]]  # gets the probabilities of sending to corresponding sharer
        return np.random.choice(a=row_labels, p=label_probabilities).item()  # converts numpy.str to python str
    # endregion

    # region resource utilization methods
    @staticmethod
    def get_resource_utilization(*args):
        """
        :param *args: Variable length argument list. See below
        :keyword arg:
        :arg 'cpu': system wide float detailing cpu usage as a percentage,
        :arg 'cpu_count': number of non-logical cpu on the machine as an int
        :arg 'cpu_avg': average system load over the last 1, 5 and 15 minutes as a tuple
        :arg 'mem': statistics about memory usage as a named tuple including the following fields (total, available), expressed in bytes as floats
        :arg 'disk': get_disk_usage dictionary with total and used keys (gigabytes as float) and percent key as float
        :return dict<str, obj> detailing the usage of the respective key arg. If arg is invalid the value will be -1.
        """
        results = {}
        for arg in args:
            results[arg] = rT.get_value(arg)
        return results
    # endregion

    # region mock methods
    def get_all_parts(self):
        """
        Sends a copy of all parts stored in this Worker instance to the requestor
        Warning: Don't use this method in production
        :returns a deep copy of self.file_parts
        :rtype dict<str, dict<int, domain.SharedFileParts>>
        """
        return deepcopy(self.sf_parts)
    # endregion
