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
    def __init_recovery_protocol(self, part):
        """
        When a corrupt file is received initiate recovery protocol, if this is the node with the most file parts
        The recovery protocol consists of reconstructing the damaged file part from other parts on the system, it may be
        necessary to obtain other files from other nodes to initiate reconstruction
        # Note to self - This is not important right now! This is only important after MCMC with metropolis hastings works
        # For now assume that when a node dies, if it had less than N-K parts, his parts are given to someone else
        """
        # TODO:
        #  future-iterations:
        #  1. corrupted or missing file recovery algorithm
        log.warning("domain.Worker.__init_recovery_protocol is only a mock. method needs to be implemented...")
    # endregion

    # region instance methods
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
        raise NotImplementedError

    def remove_file_routing(self, sf_name):
        """
        :param sf_name: a file name that is being shared on the hive which this node should stop transimitting
        :type str
        """
        raise NotImplementedError

    def receive_part(self, part, no_check=False):
        """
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
            self.__init_recovery_protocol(part)

    def receive_parts(self, sf_name, sf_id_parts, no_check=True):
        """
        :param sf_name: name of the file the parts belong to
        :type str
        :param sf_id_parts: mapping of shared file part id to SharedFileParts instances
        :type dict<int, domain.SharedFilePart>
        :param no_check: wether or not method verifies sha256 of each part.
        :type bool
        """
        if no_check:
            # Use sf_name in param to leverage the pythonic way of merging dictionaries
            self.sf_parts[sf_name].update(sf_id_parts)
        else:
            for sf_part in sf_id_parts.values():
                # When adding one 1-by-1, no other param other than the SharedFilePart is needed... See receive_part
                self.receive_part(sf_part, no_check=False)

    def send_part(self):
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

    def leave_hive(self):
        """
        Resets the field of the Worker instance, returns a deep copy of self.file_parts for hivemind convinience!
        Actual method shouldn't return anything! I repeat, this is just a shortcut! Thus is not docstringed.
        """
        sf_parts = deepcopy(self.send_shared_parts())
        self.hivemind = None
        self.sf_parts = None
        return sf_parts

    def drop_shared_file(self, shared_file_name):
        """
        Worker instance stops sharing the named file
        :param shared_file_name: the name of the file to drop from shared file structures
        """
        try:
            self.shared_files.pop(shared_file_name)
        except KeyError:
            log.error("Key ({}) doesn't exist in worker {}'s sf_parts dict".format(shared_file_name, self.name))

    def send_shared_parts(self):
        """
        :returns a deep copy of self.file_parts
        :rtype dict<str, dict<int, domain.SharedFileParts>>
        """
        return deepcopy(self.sf_parts)

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

    # region static methods
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
