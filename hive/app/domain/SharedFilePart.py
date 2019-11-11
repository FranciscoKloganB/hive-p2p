from utils import convertions, crypto


class SharedFilePart:
    """
    Represents a simulation over the P2P Network that tries to persist a file using stochastic swarm guidance
    :ivar str part_name: original name of the file this part belongs to
    :ivar int part_number: unique identifier for this file on the P2P network
    :ivar str part_id: concatenation of part_name | part_number
    :ivar str part_data: base64 string corresponding to the actual contents of this file part
    :ivar str sha256: hash value resultant of applying sha256 hash function over part_data param
    """

    # region class variables, instance variables and constructors
    def __init__(self, part_name: str, part_number: int, part_data: bytes):
        """
        Instantiates a SharedFilePart object
        :param str part_name: original name of the file this part belongs to
        :param int part_number: number that uniquely identifies this file part
        :param bytes part_data: Up to 2KB blocks of raw data that can be either strings or bytes
        """
        self.part_name: str = part_name
        self.part_number: int = part_number
        self.part_id: str = part_name + "_#_" + str(part_number)
        self.part_data: str = convertions.bytes_to_base64_string(part_data)
        self.sha256: str = crypto.sha256(self.part_data)
    # endregion

    # region override
    def __str__(self):
        return "part_name: {},\npart_number: {},\npart_id: {},\npart_data: {},\nsha256: {}\n".format(
            self.part_name, self.part_number, self.part_id, self.part_data, self.sha256
        )
    # endregion
