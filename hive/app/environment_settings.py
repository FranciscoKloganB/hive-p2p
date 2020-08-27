"""Module with simulation and project related variables.

This module demonstrates holds multiple constant variables that are used
through out the simulation's lifetime including initialization and execution.

Note:
    To configure the amount of available
    :py:class:`Network Nodes <app.domain.network_nodes.Node>` system,
    the initial size of a file
    :py:class:`Cluster Group <app.domain.cluster_groups.Cluster>` that
    work on the durability of a file, the way files are
    :py:meth:`distributed <app.domain.cluster_groups.Cluster.spread_files>`
    among the clusters' nodes at the start of a simulation and, the actual
    name of the file whose persistence is being simulated, you should create
    a simulation file using this :py:mod:`script <app.simfile_generator>` and
    follow the respective instructions. To run the script type in your
    command line terminal:

    |

    ::

        $ python simfile_generator.py --file=filename.json

    |

    It is also strongly recommended that the user does not alter any
    undocumented attributes or module variables unless they are absolutely
    sure of what they do and the consequence of their changes. These include
    variables such as :py:const:`~app.environment_settings.SHARED_ROOT` and
    :py:const:`~app.environment_settings.SIMULATION_ROOT`.
"""
import os

DEBUG: bool = False
"""Indicates if some debug related actions or prints to the terminal should 
be performed."""

# region Simulation Settings
READ_SIZE: int = 131072
"""Defines the raw size of each file block before it's wrapped in a 
:py:class:`app.domain.helpers.smart_dataclasses.FileBlockData` instance 
object. 

Some possible values include { 32KB = 32768B; 128KB = 131072B; 512KB = 524288B; 
1MB = 1048576B; 20MB = 20971520B }.
"""

MONTH_EPOCHS: int = 21600
"""Defines how many epochs (discrete time steps) a month is represented with. 
With the default value of 21600 each epoch would represent two minutes. See 
:py:meth:`~app.domain.cluster_groups.Cluster.__assign_disk_error_chance__`."""

MIN_REPLICATION_DELAY: int = 1
"""The minimum amount of epoch time steps replica file block blocks take to 
be regenerated after their are lost."""

MAX_REPLICATION_DELAY: int = 4
"""The maximum amount of epoch time steps replica file block blocks take to 
be regenerated after their are lost."""

REPLICATION_LEVEL: int = 3
"""The amount of blocks each file block has."""

MIN_CONVERGENCE_THRESHOLD: int = 0
"""The number of consecutive epoch time steps that a 
:py:class:`~app.domain.cluster_groups.HiveCluster` must converge before epochs 
start being marked with verified convergence in 
:py:attr:`app.domain.helpers.smart_dataclasses.LoggingData.convergence_set`."""

LOSS_CHANCE: float = 0.04
"""Defines the probability of a message not being delivered to a destination 
due to network link problems, in the simulation environment."""

ABS_TOLERANCE: float = 0.05
"""Defines the maximum amount of absolute positive or negative deviation that a 
current distribution :py:attr:`~app.domain.cluster_groups.HiveCluster.cv_` can 
have from the desired steady state 
:py:attr:`~app.domain.cluster_groups.HiveCluster.v_`, in order for the 
distributions to be considered equal and thus marking the epoch as convergent. 

This constant will be used by 
:py:meth:`app.domain.cluster_groups.HiveCluster.equal_distributions` along 
with a relative tolerance that is the minimum value in 
:py:attr:`~app.domain.cluster_groups.HiveCluster.v_`.
"""
# endregion

# region DO NOT ALTER THESE
# path constants
SHARED_ROOT: str = os.path.join(os.getcwd(), 'static', 'shared')
"""Path to the folder where files to be persisted during the simulation are 
located."""

OUTFILE_ROOT: str = os.path.join(os.getcwd(), 'static', 'outfiles')
"""Path to the folder where simulation output files are located will be 
stored."""

SIMULATION_ROOT: str = os.path.join(os.getcwd(), 'static', 'simfiles')
"""Path to the folder where simulation files to be executed by 
:py:mod:`app.hive_simulation` are located."""

MATLAB_DIR: str = os.path.join(os.getcwd(), 'scripts', 'matlab')
"""Path the folder where matlab scripts are located. Used by 
:py:class:`app.domain.helpers.matlab_utils.MatlabEngineContainer`"""

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
