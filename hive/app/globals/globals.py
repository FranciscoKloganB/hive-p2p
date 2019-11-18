import os

DEBUG: bool = True

# region integer constants
READ_SIZE: int = 2048
MIN_CONVERGENCE_THRESHOLD: int = 5
DEFAULT_COLUMN: int = 0
# endregion

# region float constants
A_TOL: float = 1e-2
R_TOL: float = 0.4
# endregion

# region path constants
SHARED_ROOT: str = os.path.join(os.getcwd(), 'static', 'shared')
OUTFILE_ROOT: str = os.path.join(os.getcwd(), 'static', 'outfiles')
SIMULATION_ROOT: str = os.path.join(os.getcwd(), 'static', 'simfiles')
# endregion
