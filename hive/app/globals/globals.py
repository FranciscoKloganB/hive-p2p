"""Global Simulation Settings.

This module demonstrates holds multiple constant variables that are used
through out the simulation's lifetime including initialization and execution.

Note:
    To configure the amount of available network nodes in a simulation (
    :py:class:`~domain.Worker.Worker`), the number of network nodes in a group
    persisting a file (:py:class:`~domain.Hive.Hive`), the way files are
    initially distributed between network nodes of a simulation (
    :py:meth:`~domain.Hive.Hive.spread_files`) and, the actual name of the
    file whose persistence is being simulated, you should create a simulation
    file using :py:mod:`simulation_file_generator` and follow its
    instructions. To run that modules functionality use::

        $ python simfile_generator.py --file=filename.json

    It is also strongly recommended that the user does not alter any
    undocumented attributes or module variables unless they are absolutely
    sure of what they do and the consequence of their changes. These include
    variables such as `SHARED_ROOT` and `SIMULATION_ROOT`.

Attributes:
    DEBUG:
        Indicates if some debug related actions or prints to the terminal
        should be performed (default is False)
    READ_SIZE:
        Defines the raw size of each file block before it's wrapped in a
        :py:class:`~domain.helpers.SharedFilePart.SharedFilePart` instance
        object (default is 131072). Example values: 32KB = 32768b;
        128KB = 131072b; 512KB = 524288b; 20MB = 20971520b.
    MIN_DETECTION_DELAY:
        The minimum amount of epoch time steps replica file block replicas
        take to be regenerated after their are lost (default is 1)
    MAX_DETECTION_DELAY:
        The maximum amount of epoch time steps replica file block replicas
        take to be regenerated after their are lost (default is 4)
    REPLICATION_LEVEL:
        The amount of replicas each file block has (default is 3)
    MIN_CONVERGENCE_THRESHOLD:
        The number of consecutive epoch time steps that an
        :py:class:`~domain.Hive.Hive` must converge before epochs start being
        marked with verified convergence in
        :py:attr:`~domain.helpers.SimulationData.SimulationData.convergence_set`
         (default is 2).
    LOSS_CHANCE:
        Defines the probability of a message not being delivered to a
        destination due to network link problems, in the simulation environment.
    ABS_TOLERANCE:
        Defines the maximum amount of absolute positive or negative deviation
        that a current distribution :py:func:`~domain.Hive.Hive.cv_` can have
        from the desired steady state :py:func:`~domain.Hive.Hive.v_`,
        in order for the distributions to be considered equal and thus
        marking the epoch as being in convergence in
        :py:attr:`~domain.helpers.SimulationData.SimulationData
        .convergence_set`. This constant will be used by
        :py:meth:`~domain.Hive.Hive.equal_distributions` along with a
        relative tolerance that is the minimum value in
        :py:func:`~domain.Hive.Hive.v_`.
"""

import os

DEBUG: bool = False

# region Simulation Settings

READ_SIZE: int = 131072

MIN_DETECTION_DELAY: int = 1
MAX_DETECTION_DELAY: int = 4
REPLICATION_LEVEL: int = 3
MIN_CONVERGENCE_THRESHOLD: int = 2
LOSS_CHANCE: float = 0.04
ABS_TOLERANCE: float = 0.05
# endregion

# region DO NOT ALTER THESE

# path constants
SHARED_ROOT: str = os.path.join(os.getcwd(), 'static', 'shared')
OUTFILE_ROOT: str = os.path.join(os.getcwd(), 'static', 'outfiles')
SIMULATION_ROOT: str = os.path.join(os.getcwd(), 'static', 'simfiles')
MATLAB_DIR: str = os.path.join(os.getcwd(), 'scripts', 'matlabscripts')

# Others
TRUE_FALSE = [True, False]
DELIVER_CHANCE: float = 1.0 - LOSS_CHANCE
COMMUNICATION_CHANCES = [LOSS_CHANCE, DELIVER_CHANCE]

# endregion
