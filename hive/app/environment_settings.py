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
from typing import List

from utils.convertions import truncate_float_value
import numpy

OPTIMIZE: bool = True

DEBUG: bool = True
"""Indicates if some debug related actions or prints to the terminal should 
be performed."""

BLOCKS_SIZE: int = 1 * 1024 * 1024
"""Defines the raw size of each file block before it's wrapped in a 
:py:class:`~app.domain.helpers.smart_dataclasses.FileBlockData` instance 
object. 

Some possible values include { 32KB = 32768B; 128KB = 131072B; 512KB = 524288B; 
1MB = 1048576B; 20MB = 20971520B }.
"""


def set_blocks_size(n: int) -> None:
    """Changes :py:const:`BLOCKS_SIZE` constant value at run time to the given n bytes."""
    global BLOCKS_SIZE
    BLOCKS_SIZE = n


BLOCKS_COUNT: int = 333
"""Defines into how many 
:py:class:`~app.domain.helpers.smart_dataclasses.FileBlockData` instances a file
is divided into. Either use this or :py:const:`BLOCKS_SIZE` but not both."""


def set_blocks_count(n: int) -> None:
    """Changes :py:const:`BLOCKS_COUNT` constant value at run time."""
    global BLOCKS_COUNT
    BLOCKS_COUNT = n


NEWSCAST_CACHE_SIZE: int = 20
"""The maximum amount of neighbors a :py:attr:`NewscastNode view 
<app.domain.network_nodes.NewscastNode>` can have at any given time."""

MONTH_EPOCHS: int = 21600
"""Defines how many epochs (discrete time steps) a month is represented with. 
With the default value of 21600 each epoch would represent two minutes. See 
:py:func:`~get_disk_error_chances`."""

MIN_REPLICATION_DELAY: int = 1
"""The minimum amount of epoch time steps replica file block blocks take to 
be regenerated after their are lost."""

MAX_REPLICATION_DELAY: int = 3
"""The maximum amount of epoch time steps replica file block blocks take to 
be regenerated after their are lost."""

REPLICATION_LEVEL: int = 3
"""The amount of replicas each file block has."""


def set_replication_level(n: int) -> None:
    """Changes :py:const:`REPLICATION_LEVEL` constant value at run time."""
    global REPLICATION_LEVEL
    REPLICATION_LEVEL = n


MIN_CONVERGENCE_THRESHOLD: int = 0
"""The number of consecutive epoch time steps that a 
:py:class:`~app.domain.cluster_groups.SGCluster` must converge before epochs 
start being marked with verified convergence in 
:py:attr:`app.domain.helpers.smart_dataclasses.LoggingData.convergence_set`."""

LOSS_CHANCE: float = 0.04
"""Defines the probability of a message not being delivered to a destination 
due to network link problems, in the simulation environment."""

DELIVER_CHANCE: float = 1.0 - LOSS_CHANCE
"""Defines the probability of a message being delivered to a destination, 
in the simulation environment."""

COMMUNICATION_CHANCES = [LOSS_CHANCE, DELIVER_CHANCE]


def set_loss_chance(v: float) -> None:
    """Changes :py:const:`LOSS_CHANCE` constant value at run time."""
    global LOSS_CHANCE
    global DELIVER_CHANCE
    global COMMUNICATION_CHANCES
    LOSS_CHANCE = numpy.clip(v, 0.0, 1.0)
    DELIVER_CHANCE = 1.0 - LOSS_CHANCE
    COMMUNICATION_CHANCES = [LOSS_CHANCE, DELIVER_CHANCE]


ATOL: float = 0.05
"""Defines the maximum amount of absolute positive or negative deviation that a 
current distribution :py:attr:`~app.domain.cluster_groups.SGCluster.cv_` can 
have from the desired steady state 
:py:attr:`~app.domain.cluster_groups.SGCluster.v_`, in order for the 
distributions to be considered equal and thus marking the epoch as convergent. 

This constant will be used by 
:py:meth:`app.domain.cluster_groups.SGCluster.equal_distributions` along 
with a relative tolerance that is the minimum value in 
:py:attr:`~app.domain.cluster_groups.SGCluster.v_`.
"""

RTOL: float = 0.05
"""Defines the maximum amount of relative positive or negative deviation that a 
current distribution :py:attr:`~app.domain.cluster_groups.SGCluster.cv_` can 
have from the desired steady state 
:py:attr:`~app.domain.cluster_groups.SGCluster.v_`, in order for the 
distributions to be considered equal and thus marking the epoch as convergent. 

This constant will be used by 
:py:meth:`app.domain.cluster_groups.SGCluster.equal_distributions` along 
with a relative tolerance that is the minimum value in 
:py:attr:`~app.domain.cluster_groups.SGCluster.v_`.
"""


def get_disk_error_chances(simulation_epochs: int) -> List[float]:
    """Defines the probability of a file block being corrupted while stored
    at the disk of a :py:class:`network node <app.domain.network_nodes.Node>`.

    Note:
        Recommended value should be based on the paper named
        `An Analysis of Data Corruption in the Storage Stack
        <http://www.cs.toronto.edu/bianca/papers/fast08.pdf>`_. Thus
        the current implementation follows this formula:

            (:py:const:`~app.domain.master_servers.Master.MAX_EPOCHS` / :py:const:`MONTH_EPOCHS`) * ``P(Xt ≥ L)``)

        The notation ``P(Xt ≥ L)`` denotes the probability of a disk
        developing at least L checksum mismatches within T months since
        the disk’s first use in the field. As described in linked paper.

    Args:
        simulation_epochs:
            The number of epochs the simuulation is expected to run
            assuming no failures occur.

    Returns:
        A two element list with respectively, the probability of losing
        and the probability of not losing a file block due to disk
        errors, at an epoch basis.
    """
    ploss_month = 0.0086
    ploss_epoch = (simulation_epochs * ploss_month) / MONTH_EPOCHS
    ploss_epoch = truncate_float_value(ploss_epoch, 6)
    return [ploss_epoch, 1.0 - ploss_epoch]


# region Other simulation constants
TRUE_FALSE = [True, False]
# endregion

# region OS paths
SHARED_ROOT: str = os.path.join(os.getcwd(), 'static', 'shared')
"""Path to the folder where files to be persisted during the simulation are 
located."""

SIMULATION_ROOT: str = os.path.join(os.getcwd(), 'static', 'simfiles')
"""Path to the folder where simulation files to be executed by 
:py:mod:`app.hive_simulation` are located."""

OUTFILE_ROOT: str = os.path.join(os.getcwd(), 'static', 'outfiles')
"""Path to the folder where simulation output files are located."""

RESOURCES_ROOT: str = os.path.join(os.getcwd(), 'static', 'resources')
"""Path to the folder where miscellaneous files are located."""

MIXING_RATE_SAMPLE_ROOT: str = os.path.join(OUTFILE_ROOT, 'mixing_rate_samples')

MATLAB_DIR: str = os.path.join(os.getcwd(), 'scripts', 'matlab')
"""Path the folder where matlab scripts are located. Used by 
:py:class:`~app.domain.helpers.matlab_utils.MatlabEngineContainer`"""
# endregion

# region Module paths
MASTER_SERVERS: str = 'domain.master_servers'
CLUSTER_GROUPS: str = 'domain.cluster_groups'
NETWORK_NODES: str = 'domain.network_nodes'
# endregion
