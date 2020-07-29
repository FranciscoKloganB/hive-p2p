from __future__ import annotations

from random import randint
from utils import convertions, crypto
from globals.globals import REPLICATION_LEVEL, MIN_DETECTION_DELAY, MAX_DETECTION_DELAY


class SharedFilePart:
    """Wrapping class for the contents of a file block.

    Among other responsabilities SharedFilePart helps managing simulation
    parameters, e.g., replica control such or file block integrity.

    Attributes:
        hive_id:
            Unique identifier of the hive that manages the shared file part.
        name:
            The name of the file the file block belongs to.
        number:
            The number that uniquely identifies the file block.
        id:
            Concatenation the the `name` and `number`.
        references:
            Tracks how many references exist to the file block in the
            simulation environment. When it reaches 0 the file block ceases
            to exist and the simulation fails.
        recovery_epoch:
            When a reference to the file block is lost, i.e., decremented,
            a recovery_epoch that simulates failure detection and recovery
            delay is assigned to this attribute. Until a loss occurs and
            after a loss is recovered, `recovery_epoch` is set to positive
            infinity.
        data:
            A base64-encoded string representation of the file block bytes.
        sha256:
            The hash value of data resulting from a SHA256 digest.
    """

    def __init__(
            self, hive_id: str, name: str, number: int, data: bytes
    ) -> None:
        """Creates an instance of SharedFilePart

        Args:
            hive_id:
                Unique identifier of the hive that manages the shared file part.
            name:
                The name of the file the file block belongs to.
            number:
                The number that uniquely identifies the file block.
            data:
                Actual file block data as a sequence of bytes.
        """
        self.hive_id = hive_id
        self.name: str = name
        self.number: int = number
        self.id: str = name + "_#_" + str(number)
        self.references: int = 0
        self.recovery_epoch: float = float('inf')
        self.data: str = convertions.bytes_to_base64_string(data)
        self.sha256: str = crypto.sha256(self.data)

    # region Simulation Interface
    def set_recovery_epoch(self, epoch: int) -> int:
        """Sets the epoch in which replication levels should be restored.

        This method tries to assign a new epoch, in the future, at which
        recovery should be performed. If the proposition is sooner than the
        previous proposition then assignment is accepted, else, it's rejected.

        Note:
            This method of calculating the `recovery_epoch` may seem
            controversial, but the justification lies in the assumption that
            if there are more network nodes monitoring file parts,
            than failure detections should be in theory, faster, unless
            complex consensus algorithms are being used between volatile
            peers, which is not our case. We assume peers only report their
            suspicions to a small number of trusted of monitors who then
            decide if the reported network node is disconnected, consequently
            losing the instance of SharedFilePart and possibly others.

        Args:
            epoch:
                Simulation's current epoch.

        Returns:
            Zero if the current `recovery_epoch` is positive infinity,
            otherwise the expected delay is returned. This value can be
            used to log, for example, the average recovery delay in the
            Hive simulation.
        """
        new_proposed_epoch = float(epoch + randint(MIN_DETECTION_DELAY, MAX_DETECTION_DELAY))
        if new_proposed_epoch < self.recovery_epoch:
            self.recovery_epoch = new_proposed_epoch
        return 0 if self.recovery_epoch == float('inf') else self.recovery_epoch - float(epoch)

    def update_epochs_to_recover(self, epoch: int) -> None:
        """Update the `recovery_epoch` after a recovery attempt was carried out.

        If the recovery attempt performed by some network node successfully
        managed to restore the replication levels to the original target, then,
        `recovery_epoch` is set to positive infinity, otherwise, another
        attempt will be done in the next epoch.

        Args:
            epoch:
                Simulation's current epoch.
        """
        self.recovery_epoch = float('inf') if self.references == REPLICATION_LEVEL else float(epoch + 1)

    def can_replicate(self, epoch: int) -> int:
        """Informs the calling network node if file block needs replication.

        Args:
            epoch:
                Simulation's current epoch.

        Returns:
            How many times the caller should replicate the block. The network
            node knows how many replicas he needs to create and distribute if
            returned value is bigger than zero.
        """
        if self.recovery_epoch == float('inf'):
            return 0

        if 0 < self.references < REPLICATION_LEVEL and self.recovery_epoch - float(epoch) <= 0.0:
            return REPLICATION_LEVEL - self.references

        return 0
    # endregion

    # region Overrides
    def __str__(self):
        """Overrides default string representation of SharedFilePart instances.

        Returns:
            A dictionary representation of the object.
        """
        return "part_name: {},\npart_number: {},\npart_id: {},\npart_data: {},\nsha256: {}\n".format(self.name, self.number, self.id, self.data, self.sha256)
    # endregionss

    # region Helpers

    def decrement_and_get_references(self):
        """Decreases by one and gets the number of file block references

        Returns:
            The number of file block references existing in the simulation
            environment.
        """
        self.references -= 1
        return self.references
    
    # endregion
