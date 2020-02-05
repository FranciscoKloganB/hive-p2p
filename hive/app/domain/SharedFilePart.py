from random import randint
from utils import convertions, crypto
from globals.globals import REPLICATION_LEVEL, MIN_DETECTION_DELAY, MAX_DETECTION_DELAY


class SharedFilePart:
    """
    Represents a simulation over the P2P Network that tries to persist a file using stochastic swarm guidance
    :ivar str id: concatenation of part_name | part_number
    :ivar str hive_id: uniquely identifies the hive that manages the shared file part instance
    :ivar str name: original name of the file this part belongs to
    :ivar int number: unique identifier for this file on the P2P network
    :ivar int references: indicates how many references exist for this SharedFilePart
    :ivar int epochs_to_recover: indicates when recovery of this file will occur during
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
        self.references: int = REPLICATION_LEVEL
        self.epochs_to_recover: int = -1
        self.data: str = convertions.bytes_to_base64_string(data)
        self.sha256: str = crypto.sha256(self.data)
    # endregion

    # region Simulation Interface
    def set_epochs_to_recover(self) -> None:
        """
        When epochs_to_recover is a negative number (usually -1), it means that at least one reference to the SharedFilePart was lost in the current epoch; in
        this case, set_recovery_delay assigns a number of epochs until a Worker who posses one reference to the SharedFilePart instance can generate references
        for some other Workers.
        """
        if self.epochs_to_recover < 0:
            self.epochs_to_recover = randint(MIN_DETECTION_DELAY, MAX_DETECTION_DELAY)

    def reset_epochs_to_recover(self) -> None:
        self.epochs_to_recover = -1

    def need_to_replicate_part(self) -> bool:
        return True if self.references < REPLICATION_LEVEL and self.epochs_to_recover == 0 else False
    # endregion

    # region Overrides
    def __str__(self):
        return "part_name: {},\npart_number: {},\npart_id: {},\npart_data: {},\nsha256: {}\n".format(self.name, self.number, self.id, self.data, self.sha256)
    # endregionss

    # region Helpers
    def decrease_and_get_references(self):
        self.references = self.references - 1
        return self.references

    def increase_and_get_references(self):
        self.references = self.references + 1
        return self.references
    # endregion
