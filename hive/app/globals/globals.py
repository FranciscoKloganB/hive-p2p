import os

DEBUG: bool = True

# region integer constants
READ_SIZE: int = 8192
DEFAULT_COLUMN: int = 0
REPLICATION_LEVEL: int = 3
MIN_CONVERGENCE_THRESHOLD: int = 3
# endregion

# region float constants
AVG_UPTIME: float = 0.4
A_TOL: float = 1e-2
R_TOL: float = 0.4
# endregion

# region path constants
SHARED_ROOT: str = os.path.join(os.getcwd(), 'static', 'shared')
OUTFILE_ROOT: str = os.path.join(os.getcwd(), 'static', 'outfiles')
SIMULATION_ROOT: str = os.path.join(os.getcwd(), 'static', 'simfiles')
# endregion
