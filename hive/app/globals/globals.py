import os

DEBUG: bool = False

# region integer constants
READ_SIZE: int = 2048
MIN_CONVERGENCE_THRESHOLD: int = 5
DEFAULT_COLUMN: int = 0
# endregion

# region path constants
SHARED_ROOT: str = os.path.join(os.getcwd(), 'static', 'shared')
OUTFILE_ROOT: str = os.path.join(os.getcwd(), 'static', 'outfiles')
SIMULATION_ROOT: str = os.path.join(os.getcwd(), 'static', 'simfiles')
# endregion
