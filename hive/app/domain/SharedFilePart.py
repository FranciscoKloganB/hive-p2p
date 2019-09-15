from utils import ConvertUtils, CryptoUtils


class SharedFilePart:
    """
    Represents a simulation over the P2P Network that tries to persist a file using stochastic swarm guidance

    :param :ivar part_id: the name of the shared file without extensions concatenated with a number for uniqueness
    :type str
    :param :ivar part_data: arbitrary string or binary data that constitutes the part content
    :type str

    :ivar sha256: hash value resultant of applying sha256 hash function over part_data param
    :type str
    """

    def __init__(self, part_id, part_data):
        self.part_id = part_id
        self.part_data = ConvertUtils.bytes_to_base64_string(part_data)
        self.sha256 = CryptoUtils.sha256(part_data)
