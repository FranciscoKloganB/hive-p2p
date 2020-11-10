Quickstart
==========

.. toctree::
   :maxdepth: 3
   :caption: Quickstart
   :name: Quickstart

Technology
----------

This simulator uses Python 3.7.7. You are free to use any version you desire,
but we do not guarantee the simulator will work under such conditions.
Any version launched before 3.7.x will not run this project due to retro
compatibility errors. We recommend using an IDE such as PyCharm or equivalent
for easier code inspection, usage and overall faster and, more stable workflows.

Installation - Part I
~~~~~~~~~~~~~~~~~~~~~

1. Download and install Python 3.7.x or higher:

|
   - https://www.python.org/downloads/release/python-377/

2. Clone our repository at:

|
   - https://github.com/FranciscoKloganB/hivessimulator

3. Set a Python environment variable at:

|
   - ``hivessimulator/app``

4. We recommended using JetBrains' IDEs, but you can skip this step:

|
   - https://www.jetbrains.com/pycharm/download

5. Create a virtual environment of your choosing, two example guides are
linked below:

|
   - https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
|
   - https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html

6. Navigate to ``hive`` folder located at the root of your recently cloned
project:

|
   - ``$ cd hive``

7. Install project dependencies by opening your terminal and inserting the
command:

|
   - ``$ pip install -r requirements.txt``

The previous steps complete the setup of your Hives project. If you have or
can obtain licenses for Mosek_ or MatLab_, read `Installation - Part II`_,
otherwise read `Disabling Licensed Components`_.

Installation - Part II
~~~~~~~~~~~~~~~~~~~~~~

Throughout the development of the project, a handful of convex optimization
problems had to be solved. We used the CVXPY_ package to tackle that issue;
unfortunately, the solvers available to Python_ are few and not very powerful,
specially when it comes to open-source ones. For semi-definite programming
problems with utilized Mosek_. We let CVXPY select the solver for global
optimization problems, from among the pool of installed solvers. We also use
BMIBNB_ solver from MatLab_ (through the MatLabEngine_) using YALMIP_ because
the latter supports non-convex constraints.

To use MOSEK along with CVXPY follow the installation instructions linked below:

1. Mosek licensing quick start.

|
   - https://docs.mosek.com/9.2/licensing/quickstart.html

2. Installing Mosek on your python environment.

|
   - https://docs.mosek.com/9.2/pythonapi/install-interface.html

For the MatLab Engine to work you need to have
`MatLab R2020a <https://www.mathworks.com/products/new_products/latest_features.html>`_
or higher installed on your machine with a valid license. After you installing
and validating the software, you should
`install YALMIP <https://yalmip.github.io/tutorial/installation/>`_. BMIBNB
is bundled with YALMIP by default and no further action is required.

.. _CVXPY: https://www.cvxpy.org/

.. _Python: https://www.python.org/

.. _Mosek: https://www.mosek.com/products/academic-licenses/

.. _MatLab: https://https://www.mathworks.com/products/matlab.html

.. _BMIBNB: https://yalmip.github.io/solver/bmibnb/

.. _YALMIP: https://yalmip.github.io/

.. _MatLabEngine: https://www.mathworks.com/help/matlab/matlab-engine-for-python.html


Disabling Licensed Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As explained in the previous session, some licensed components were needed
during the development of our research, which we bundled with the simulator
source code for demonstrative purposes. Concerning Mosek_, you do not need to
take any action. Any modules' functions that use the
`Mosek Optimizer API <https://docs.mosek.com/9.2/pythonapi/index.html>`_
through CVXPY_ check if the package is installed and properly licensed before
using it in favor of other open-source solvers. Concerning MatLab_ you should
not need any further action either - our modules deal both with invalid
licenses and Pythons'
`AttributeError <https://docs.python.org/3.7/library/exceptions.html#AttributeError>`_
transparently when invoking MatLabEngine_ methods as a result of our singleton,
thread-safe, implementation of :py:class:`~app.domain.helpers.matlab_utils.MatlabEngineContainer`.

Usage
-----

A typical usage of the Hives simulator would include the following sequence of
commands (see :ref:`Scripts and Flags` section for option details), responding
accordingly to any prompts that appear on your command line terminal:

   ``$ cd hive/app``

   ``$ python simfile_generator.py --file=test01.json``

   ``$ python hive_simulation.py --file=test01.json --iters=30 --epochs=720``
