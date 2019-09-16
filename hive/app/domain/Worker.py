from utils.ResourceTracker import ResourceTracker as rT

class Worker:
    """
    Defines a node on the P2P network. Workers are subject to constraints imposed by Hiveminds, constraints they inflict
    on themselves based on available computing power (CPU, RAM, etc...) and can have [0, N] shared file parts. Workers
    have the ability to reconstruct lost file parts whene needed.
    :ivar shared_file_parts: part_name is a key to a dict of integer part_id keys leading to actual SharedFileParts
    :type dict<string, dict<int, SharedFilePart>>
    """

    def __init__(self):
        self.shared_file_parts = {}

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

