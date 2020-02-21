import os

DEBUG: bool = True

# region Simulation Settings
READ_SIZE: int = 32768  # 32KB blocks. Should wield ~1440 SharedFileParts if the original file is ~45MB in size. With REPLICATION_LEVEL = 3, we run with 4320 parts
MAX_EPOCHS = 720  # One day has 24h, meaning that one epoch per minute wwould be 1440, 720 defines one epoch every two minutes
MAX_EPOCHS_PLUS = MAX_EPOCHS + 1
MIN_DETECTION_DELAY: int = 1  # 2 minutes
MAX_DETECTION_DELAY: int = 5  # 10 minutes
REPLICATION_LEVEL: int = 3  # Each file part has 3 copies, for simulation purposes, this copies are soft copies.
MIN_CONVERGENCE_THRESHOLD: int = 3
LOSS_CHANCE: float = 0.04  # Each sent file as a 4% chance of timing out due to message being lost in travel
DELIVER_CHANCE: float = 1.0 - LOSS_CHANCE  # Each sent file as a 4% chance of timing out due to message being lost in travel
# endregion

# region Integer Constants
DEFAULT_COLUMN: int = 0
# endregion

# region Float Constants
A_TOL: float = 1e-2
R_TOL: float = 0.4
# endregion

# region Path Constants
SHARED_ROOT: str = os.path.join(os.getcwd(), 'static', 'shared')
OUTFILE_ROOT: str = os.path.join(os.getcwd(), 'static', 'outfiles')
SIMULATION_ROOT: str = os.path.join(os.getcwd(), 'static', 'simfiles')
# endregion

# region Other Constants
TRUE_FALSE = [True, False]
COMMUNICATION_CHANCES = [LOSS_CHANCE, DELIVER_CHANCE]
HIVE_SIZE_BEFORE_RECOVER = "sizeBeforeRecover"
HIVE_SIZE_AFTER_RECOVER = "sizeAfterRecover"
HIVE_STATUS_BEFORE_RECOVER = "statusBeforeRecover"
HIVE_STATUS_AFTER_RECOVER = "statusAfterRecover"
EPOCH_RECOVERY_DELAY = "epochAvgDelay"
# endregion
