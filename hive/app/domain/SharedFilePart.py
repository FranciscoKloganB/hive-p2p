from utils import convertions, crypto


class SharedFilePart:
    """
    Represents a simulation over the P2P Network that tries to persist a file using stochastic swarm guidance
    :ivar str id: concatenation of part_name | part_number
    :ivar str hive_id: uniquely identifies the hive that manages the shared file part instance
    :ivar str name: original name of the file this part belongs to
    :ivar int number: unique identifier for this file on the P2P network
    :ivar str data: base64 string corresponding to the actual contents of this file part
    :ivar str sha256: hash value resultant of applying sha256 hash function over part_data param
    """

    # region Class Variables, Instance Variables and Constructors
    def __init__(self, hive_id: str, name: str, number: int, data: bytes):
        """
        Instantiates a SharedFilePart object
        :param str name: original name of the file this part belongs to
        :param int number: number that uniquely identifies this file part
        :param bytes data: Up to 2KB blocks of raw data that can be either strings or bytes
        """
        self.id: str = name + "_#_" + str(number)
        self.hive_id = hive_id
        self.name: str = name
        self.number: int = number
        self.data: str = convertions.bytes_to_base64_string(data)
        self.sha256: str = crypto.sha256(self.data)
    # endregion

    # region Override
    def __str__(self):
        return "part_name: {},\npart_number: {},\npart_id: {},\npart_data: {},\nsha256: {}\n".format(self.name, self.number, self.id, self.data, self.sha256)
    # endregion
