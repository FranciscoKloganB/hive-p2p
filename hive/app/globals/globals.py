import os

DEBUG: bool = True

# region Simulation Settings
READ_SIZE: int = 32768  # 32KB blocks. Should wield ~1440 SharedFileParts if the original file is ~45MB in size. With REPLICATION_LEVEL = 3, we run with 4320 parts
MAX_EPOCHS = 720  # One day has 24h, meaning that one epoch per minute wwould be 1440, 720 defines one epoch every two minutes
MIN_DETECTION_DELAY: int = 1  # 2 minutes
MAX_DETECTION_DELAY: int = 7  # 14 minutes
AVG_UPTIME: float = 0.4
REPLICATION_LEVEL: int = 3  # Each file part has 3 copies, for simulation purposes, this copies are soft copies.
MIN_CONVERGENCE_THRESHOLD: int = 3

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
