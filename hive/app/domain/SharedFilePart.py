from utils import convertions, crypto


class SharedFilePart:
    # region docstrings
    """
    Represents a simulation over the P2P Network that tries to persist a file using stochastic swarm guidance
    :ivar part_name: original name of the file this part belongs to
    :type str
    :ivar part_number: unique identifier for this file on the P2P network
    :type int
    :ivar part_id: concatenation of part_name | part_number
    :type str
    :ivar part_data: base64 string corresponding to the actual contents of this file part
    :type str
    :ivar sha256: hash value resultant of applying sha256 hash function over part_data param
    :type str
    """
    # endregion

    # region class variables, instance variables and constructors
    def __init__(self, part_name, part_number, part_data):
        """
        :param part_name: original name of the file this part belongs to
        :type str
        :param part_number: number that uniquely identifies this file part
        :type int
        :param part_data: Up to 2KB blocks of raw data that can be either strings or bytes
        :type bytes or str
        """
        self.__part_name = part_name
        self.__part_number = part_number
        self.__part_id = part_name + "_#_" + str(part_number)
        self.__part_data = convertions.bytes_to_base64_string(part_data)
        self.__sha256 = crypto.sha256(part_data)
    # endregion

    # region properties
    @property
    def part_name(self):
        return self.__part_name

    @property
    def part_number(self):
        return self.__part_number

    @property
    def part_id(self):
        return self.__part_id

    @property
    def part_data(self):
        return self.__part_data

    @property
    def sha256(self):
        return self.__sha256
    # endregion

    # region override
    def __str__(self):
        return "part_name: {},\n part_number: {},\npart_id: {},\n part_data: {},\n sha256: {}\n".format(
            self.part_name, self.part_number, self.part_id, self.part_data, self.sha256
        )
    # endregion
