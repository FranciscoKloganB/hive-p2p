import os

DEBUG: bool = False

# region Simulation Settings
READ_SIZE: int = 131072  # 32KB = 32768b || 128KB = 131072b || 512KB = 524288b || 20MB = 20971520b. Defines the raw size of each SharedFilePart.
MAX_EPOCHS = 720  # One day has 24h, meaning that one epoch per minute wwould be 1440, 720 defines one epoch every two minutes
MAX_EPOCHS_PLUS = MAX_EPOCHS + 1
MIN_DETECTION_DELAY: int = 1  # 2 minutes
MAX_DETECTION_DELAY: int = 4  # 8 minutes
REPLICATION_LEVEL: int = 3  # Each file part has 3 copies, for simulation purposes, this copies are soft copies.
MIN_CONVERGENCE_THRESHOLD: int = 2
LOSS_CHANCE: float = 0.04  # Each sent file as a 4% chance of timing out due to message being lost in travel
DELIVER_CHANCE: float = 1.0 - LOSS_CHANCE  # Each sent file as a 4% chance of timing out due to message being lost in travel
# endregion

# region Integer Constants
DEFAULT_COL: int = 0
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
