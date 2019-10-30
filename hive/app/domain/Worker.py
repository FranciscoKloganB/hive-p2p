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
        #  corrupted or missing file recovery algorithm
        pass
    # endregion

    # region instance methods
    def set_file_routing(self, file_name, labeled_transition_vector):
        """
        :param file_name: a file name that is being shared on the hive
        :type str
        :param labeled_transition_vector: probability vector indicating transitions to other states for the given file
        :type 1-D numpy.Array in column format
        """
        self.__routing_table[file_name] = labeled_transition_vector

    def receive_part(self, part, no_check=False):
        if no_check or crypto.sha256(part.part_data) == part.sha256:
            if part.name in self.sf_parts:
                self.sf_parts[part.name][part.part_id] = part
            else:
                self.sf_parts[part.name] = {}
                self.sf_parts[part.name][part.part_id] = part
        else:
            print("part_name: {}, part_number: {} - corrupted".format(part.part_name, str(part.part_number)))
            self.__init_recovery_protocol(part)

    def send_part(self):
        for part_name, part_id_sfp_dict in self.sf_parts.items():
            tmp = {}
            for part_id, sfp_obj in part_id_sfp_dict.items():
                dest_worker = self.get_next_state(file_name=part_name)
                if dest_worker == self.name:
                    tmp[part_id] = sfp_obj
                else:
                    response_code = self.hivemind.route_file_part(dest_worker, sfp_obj)
                    if response_code != HttpCodes.OK:
                        # TODO:
                        #  make use of the HttpCode responses with more than a binary behaviour
                        tmp[part_id] = sfp_obj
            self.sf_parts[part_name] = tmp

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
        self.shared_files.pop(shared_file_name, None)

    def send_shared_parts(self):
        """
        :returns a deep copy of self.file_parts
        :rtype dict<str, dict<int, domain.SharedFileParts>>
        """
        return deepcopy(self.sf_parts)

    def get_next_state(self, file_name):
        """
        :param file_name: the name of the file the part to be routed belongs to
        :type: str
        :return: the name of the worker to whom the file should be routed too
        :type: str
        """
        routing_data = self.__routing_table[file_name]
        row_labels = [*routing_data.index.values]  # gets the names of sharers as a list
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
