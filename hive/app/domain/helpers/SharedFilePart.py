from __future__ import annotations

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
    :ivar float recovery_epoch: indicates when recovery of this file will occur during
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
        self.references: int = 0
        self.recovery_epoch: float = float('inf')
        self.data: str = convertions.bytes_to_base64_string(data)
        self.sha256: str = crypto.sha256(self.data)
    # endregion

    # region Simulation Interface
    def set_recovery_epoch(self, epoch: int) -> int:
        """
        Assigns a value to the instance's recovery_epoch attribute that indicates when a Worker who posses a reference to it, can replicate the part.
        :param int epoch: current simulation's epoch
        :returns int: expected delay
        """
        new_proposed_epoch = float(epoch + randint(MIN_DETECTION_DELAY, MAX_DETECTION_DELAY))
        if new_proposed_epoch < self.recovery_epoch:
            self.recovery_epoch = new_proposed_epoch
        return 0 if self.recovery_epoch == float('inf') else self.recovery_epoch - float(epoch)

    def reset_epochs_to_recover(self, epoch: int) -> None:
        """
        Resets self.recovery_epoch attribute back to the default value of -1
        :param int epoch: current simulation's epoch
        """
        self.recovery_epoch = float('inf') if self.references == REPLICATION_LEVEL else float(epoch + 1)

    def can_replicate(self, current_epoch: int) -> int:
        """
        :param int current_epoch: current simulation's epoch
        :returns int: how many times the caller should replicate the SharedFilePart instance, if such action is possible
        """
        if self.recovery_epoch == float('inf'):
            return 0
        elif 0 < self.references < REPLICATION_LEVEL and self.recovery_epoch - float(current_epoch) <= 0.0:
            return REPLICATION_LEVEL - self.references
        else:
            return 0
    # endregion

    # region Overrides
    def __str__(self):
        return "part_name: {},\npart_number: {},\npart_id: {},\npart_data: {},\nsha256: {}\n".format(self.name, self.number, self.id, self.data, self.sha256)
    # endregionss

    # region Helpers
    def decrease_and_get_references(self):
        self.references -= 1
        return self.references
    # endregion
