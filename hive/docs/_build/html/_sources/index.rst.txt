.. Hives- A P2P Stochastic Swarm Guidance Simulator documentation master file, created by
   sphinx-quickstart on Wed Aug 19 12:05:24 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Hives Quickstart
=============================================================================

Hives is a P2P Stochastic Swarm Guidance Simulator that facilitates research
by allowing developers to prototype P2P networks based on swarm guidance
behaviors, quickly. The simulator is written in Python (version 3.7.x) which
offers users easy access to powerful scientific libraries such as NumPy,
SciPy and Pandas, which are not readily available in languages like Java or C#
and as a result of some of the best or most well-known simulators out there.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Technology
----------

This simulator uses Python 3.7.7. You are free to use any version you desire,
but we do not guarantee the simulator will work under such conditions.
Any version launched before 3.7.x will not run this project due to retro
compatibility errors. We recommend using an IDE such as PyCharm or equivalent
for easier inspection, usage, and setups.

Installation
~~~~~~~~~~~~

1. Download and install Python 3.7.7 if you do not have it yet:

|
   - https://www.python.org/downloads/release/python-377/

2. Download (clone) our repository at:

|
   - https://github.com/FranciscoKloganB/hive-msc-thesis

3. We recommended using JetBrains' IDEs, but this is not necessary:

|
   - https://www.jetbrains.com/pycharm/download

4. Create a virtual environment of your choosing, two example guides are
linked below:

|
   - https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
|
   - https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html

5. Navigate to ``hive`` folder located at the root of your recently cloned
project:

|
   - ``$ cd hive``

6. Install project dependencies by opening your terminal and inserting the
command:

|
   - ``$ pip install -r requirements.txt``

Usage
-----

A typical usage of the Hives simulator would include the following sequence of
commands (see `Scripts and Flags`_ section for flag details), responding
accordingly to any prompts that appear on your command line terminal:

   ``$ cd hive/app``

   ``$ python simfile_generator.py --file=test01.json``

   ``$ python hive_simulation.py --file=test01.json --iters=30 --epochs=720``

Scripts and Flags
-----------------

+------------------------+------------------+-------------------+-------------------+------------------------------------------------------------------------------------------------------------+
| *simfile_generator.py* |                  |                   |                   |                                                                                                            |
+========================+==================+===================+===================+============================================================================================================+
| **Long option**        | **Short Option** | **Argument Type** | **Default Value** | **Description**                                                                                            |
+------------------------+------------------+-------------------+-------------------+------------------------------------------------------------------------------------------------------------+
| --file                 | -f               | str               | ``None``          | Creates a simulation file with the specified name in :py:const:`app.environment_settings.SIMULATION_ROOT`. |
+------------------------+------------------+-------------------+-------------------+------------------------------------------------------------------------------------------------------------+

Future Releases
---------------

In the future, we will focus on improving each simulation's thread performance,
in the current release, any P2P network with more than 16 peers and many files
can take a long time to complete due to the amount of *for* statements that
exist in the code. This is one reason why our program offers the possibility
of running multiple (different) simulations in different threads, allowing
researchers to complete more simulations, in less time, by fully utilizing the
CPU of their machines. Ideally, we would like to offer this speed up using
multi-threading and fast individual threads.




Indices and tables
==================

* :ref:`modindex`
