import numpy as np

from domain.MarkovChain import MarkovMatrix
from utils import ConvertUtils, CryptoUtils


class SharedFilePart:
    """
    Represents a simulation over the P2P Network that tries to persist a file using stochastic swarm guidance
    :ivar part_name: original name of the file this part belongs to
    :type str
    :ivar part_id: unique identifier for this file on the P2P network
    :type int
    :ivar part_data: base64 string corresponding to the actual contents of this file part
    :type str
    :ivar sha256: hash value resultant of applying sha256 hash function over part_data param
    :type str
    :ivar ddv: stochastic like list to define the desired distribution vector that this SharedFilePart is pursuing
    :type 1D line numpy.array
    :ivar markov_matrix: container object describing and implementing Markov Chain behaviour
    :type hive.domain.MarkovChain
    """

    def __init__(self, part_name, part_id, part_data, ddv=None, transition_matrix_definition=None):
        """
        :param part_name: original name of the file this part belongs to
        :type str
        :param part_id: number that uniquely identifies this file part
        :type int
        :param part_data: Up to 2KB blocks of raw data that can be either strings or bytes
        :type bytes or str
        :param ddv: stochastic like list to define the desired distribution vector that this SharedFilePart is pursuing
        :type list<float>
        :param transition_matrix_definition: tuple containing state names and the respective transition vectors
        :type tuple<list, list<lit<float>>
        """
        self.part_name = part_name
        self.part_id = part_id
        self.part_data = ConvertUtils.bytes_to_base64_string(part_data)
        self.sha256 = CryptoUtils.sha256(part_data)
        # self.desired_distribution = np.array(ddv).transpose() if ddv is not None else ddv
        self.markov_matrix = MarkovMatrix(transition_matrix_definition[0], transition_matrix_definition[1])

