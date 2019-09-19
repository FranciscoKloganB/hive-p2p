from utils import CryptoUtils
from utils.ResourceTracker import ResourceTracker as rT


class Worker:
    """
    Defines a node on the P2P network. Workers are subject to constraints imposed by Hivemind, constraints they inflict
    on themselves based on available computing power (CPU, RAM, etc...) and can have [0, N] shared file parts. Workers
    have the ability to reconstruct lost file parts when needed.
    :ivar hivemind: coordinator of the unstructured Hybrid P2P network that enlisted this worker for a Hive
    :type str
    :ivar name: id of this worker node that uniquely identifies him in the network
    :type str
    :ivar shared_file_parts: part_name is a key to a dict of integer part_id keys leading to actual SharedFileParts
    :type dict<string, dict<int, SharedFilePart>>
    """

    def __init__(self, hivemind, name):
        self.hivemind = hivemind
        self.name = name
        self.shared_file_parts = {}

    def __hash__(self):
        # allows a worker object to be used as a dictionary key
        return hash(str(self.name))

    def __eq__(self, other):
        return (self.hivemind, self.name) == (other.hivemind, other.name)

    def __ne__(self, other):
        return not(self == other)

    def receive_sfp(self, part):
        if CryptoUtils.sha256(part.part_data) == part.sha256:
            self.shared_file_parts[part.part_name][part.part_id] = part
        else:
            print("part_name: {}, part_id: {} - corrupted".format(part.part_name, part.part_id))
            self.init_recovery_protocol(part)

    def send_sfp(self):
        tmp_dict = {}
        for part_name, part_id_dict in self.shared_file_parts.items():
            for part_id, shared_file_part in part_id_dict.items():
                next_worker = shared_file_part.get_next_state(self.name)
                if next_worker == self.name:
                    tmp_dict[part_name][part_id] = shared_file_part
                else:
                    code = self.hivemind.hivemind_send_update(next_worker, shared_file_part)
                    if code != 200:
                        # TODO
                        pass
        self.shared_file_parts = tmp_dict

    def leave_hive(self, orderly=True):
        if orderly:
            self.hivemind.redistribute_parts(self.name, self.shared_file_parts)
        self.hivemind = None
        self.name = None
        self.shared_file_parts = None

    def init_recovery_protocol(self, part):
        # TODO
        pass

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



