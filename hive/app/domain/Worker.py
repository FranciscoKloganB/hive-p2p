from utils import CryptoUtils
from utils.ResourceTracker import ResourceTracker as rT
from domain.Enums import HttpCodes


class Worker:
    """
    Defines a node on the P2P network. Workers are subject to constraints imposed by Hivemind, constraints they inflict
    on themselves based on available computing power (CPU, RAM, etc...) and can have [0, N] shared file parts. Workers
    have the ability to reconstruct lost file parts when needed.
    :ivar hivemind: coordinator of the unstructured Hybrid P2P network that enlisted this worker for a Hive
    :type str
    :ivar name: id of this worker node that uniquely identifies him in the network
    :type str
    :ivar file_parts: part_id is a key to a SharedFilePart
    :type dict<string, SharedFilePart>
    """

    def __init__(self, hivemind, name):
        self.hivemind = hivemind
        self.name = name
        self.file_parts = {}

    def __hash__(self):
        # allows a worker object to be used as a dictionary key
        return hash(str(self.name))

    def __eq__(self, other):
        return (self.hivemind, self.name) == (other.hivemind, other.name)

    def __ne__(self, other):
        return not(self == other)

    def __init_recovery_protocol(self, part):
        """
        # TODO
        When a corrupt file is received initiate recovery protocol, if this is the node with the most file parts
        The recovery protocol consists of reconstructing the damaged file part from other parts on the system, it may be
        necessary to obtain other files from other nodes to initiate reconstruction
        """
        pass

    def receive_part(self, part):
        if CryptoUtils.sha256(part.part_data) == part.sha256:
            self.file_parts[part.part_id] = part
        else:
            print("part_name: {}, part_number: {} - corrupted".format(part.part_name, str(part.part_number)))
            self.__init_recovery_protocol(part)

    def send_part(self):
        tmp = {}
        for part_id, part in self.file_parts.items():
                dest_worker = part.get_next_state(self.name)
                if dest_worker == self.name:
                    tmp[part_id] = part
                else:
                    response_code = self.hivemind.simulate_transmission(dest_worker, part)
                    if response_code != HttpCodes.OK:
                        tmp[part_id] = part
        self.file_parts = tmp

    def leave_hive(self, orderly=True):
        if orderly:
            self.hivemind.redistribute_parts(self.name, self.file_parts)
        self.hivemind = None
        self.name = None
        self.file_parts = None

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



