"""Module with simulation and project related variables.

This module demonstrates holds multiple constant variables that are used
through out the simulation's lifetime including initialization and execution.

Note:
    To configure the amount of available network nodes in a simulation
    (:py:class:`app.domain.network_nodes.HiveNode`), the number of network
    nodes in a group persisting a file
    (:py:class:`app.domain.cluster_groups.Cluster`),
    the way files are initially distributed between network nodes of a
    simulation (:py:meth:`app.domain.cluster_groups.Cluster.spread_files`)
    and, the actual name of the file whose persistence is being simulated,
    you should create a simulation file using
    :py:mod:`simulation_file_generator` and follow its instructions. To run
    that modules functionality use::

        $ python simfile_generator.py --file=filename.json

    It is also strongly recommended that the user does not alter any
    undocumented attributes or module variables unless they are absolutely
    sure of what they do and the consequence of their changes. These include
    variables such as `SHARED_ROOT` and `SIMULATION_ROOT`.

Attributes:
    DEBUG:
        Indicates if some debug related actions or prints to the terminal
        should be performed.
    READ_SIZE:
        Defines the raw size of each file block before it's wrapped in a
        :py:class:`app.domain.helpers.smart_dataclasses.FileBlockData`
        instance object. Example values: 32KB = 32768B;
        128KB = 131072B; 512KB = 524288B; 1MB = 1048576B; 20MB = 20971520B.
    MONTH_EPOCHS:
        Defines how many epochs (discrete time steps) a month is represented
        with (with the default value of 21600 each epoch would represent two
        minutes. See :py:meth:`app.domain.cluster_groups
        ._assign_disk_error_chance`.
    MIN_REPLICATION_DELAY:
        The minimum amount of epoch time steps replica file block blocks
        take to be regenerated after their are lost.
    MAX_REPLICATION_DELAY:
        The maximum amount of epoch time steps replica file block blocks
        take to be regenerated after their are lost.
    REPLICATION_LEVEL:
        The amount of blocks each file block has.
    MIN_CONVERGENCE_THRESHOLD:
        The number of consecutive epoch time steps that an
        :py:class:`app.domain.cluster_groups.Cluster` must converge before
        epochs start being marked with verified convergence in
        :py:attr:`app.domain.helpers.smart_dataclasses.LoggingData
        .convergence_set`.
    LOSS_CHANCE:
        Defines the probability of a message not being delivered to a
        destination due to network link problems, in the simulation
        environment.
    ABS_TOLERANCE:
        Defines the maximum amount of absolute positive or negative deviation
        that a current distribution
        :py:attr:`app.domain.cluster_groups.Cluster.cv_` can have from the
        desired steady state :py:attr:`app.domain.cluster_groups.Cluster.v_`,
        in order for the distributions to be considered equal and thus
        marking the epoch as convergent. This constant will be used by
        :py:meth:`app.domain.cluster_groups.Cluster.equal_distributions`
        along with a relative tolerance that is the minimum value in
        :py:attr:`app.domain.cluster_groups.Cluster.v_`.

"""
import os

DEBUG: bool = False

# region Simulation Settings
READ_SIZE: int = 131072

MONTH_EPOCHS: int = 21600

MIN_REPLICATION_DELAY: int = 1
MAX_REPLICATION_DELAY: int = 4

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
MATLAB_DIR: str = os.path.join(os.getcwd(), 'scripts', 'matlab')
MIXING_RATE_SAMPLE_ROOT: str = os.path.join(
    OUTFILE_ROOT, 'mixing_rate_samples')

# module paths
MASTER_SERVERS: str = 'domain.master_servers'
CLUSTER_GROUPS: str = 'domain.cluster_groups'
NETWORK_NODES: str = 'domain.network_nodes'

# Others
TRUE_FALSE = [True, False]
DELIVER_CHANCE: float = 1.0 - LOSS_CHANCE
COMMUNICATION_CHANCES = [LOSS_CHANCE, DELIVER_CHANCE]
# endregion
