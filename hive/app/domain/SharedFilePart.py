import numpy as np

from utils import ConvertUtils, CryptoUtils


class SharedFilePart:
    """
    Represents a simulation over the P2P Network that tries to persist a file using stochastic swarm guidance
    :ivar markov_matrix: container object describing and implementing Markov Chain behaviour
    :type hive.domain.MarkovMatrix
    :ivar ddv: stochastic like list to define the desired distribution vector that this SharedFilePart is pursuing
    :type 1D line numpy.array
    :ivar part_id: unique identifier for this file on the P2P network
    :type str
    :ivar part_data: base64 string corresponding to the actual contents of this file part
    :type str
    :ivar sha256: hash value resultant of applying sha256 hash function over part_data param
    :type str
    """

    def __init__(self, markov_matrix, ddv, part_id, part_data):

        """
        :param markov_matrix: container object describing and implementing Markov Chain behaviour
        :type hive.domain.MarkovMatrix
        :param ddv: stochastic like list to define the desired distribution vector that this SharedFilePart is pursuing
        :type list<float>
        :param part_id: the name of the shared file without extensions concatenated with a number for uniqueness
        :type str
        :param part_data: Up to 2KB blocks of raw data that can be either strings or bytes
        :type bytes or str
        """
        self.markov_matrix = markov_matrix
        self.desired_distribution = np.array(ddv).transpose()
        self.part_id = part_id
        self.part_data = ConvertUtils.bytes_to_base64_string(part_data)
        self.sha256 = CryptoUtils.sha256(part_data)
