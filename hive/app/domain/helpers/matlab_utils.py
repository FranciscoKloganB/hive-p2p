"""Module with Matlab related classes."""
from __future__ import annotations

import threading
from typing import Any

import matlab.engine
import numpy as np

from environment_settings import MATLAB_DIR


class MatlabEngineContainer:
    """Singleton class wrapper containing thread safe access to a MatlabEngine.

    The purpose of this class is to provide a thread-safe way to access the
    matlab engine instance object when running simulations in threaded mode.

    Attributes:
        eng:
            A matlab engine instance object used for matrix and vector
            optimization operations throughout the simulations.
            Do not access this directly. MatlabEngine objects are not
            officially thread safe, thus it is recommended that you utilize
            the wrapped function, unless you are not running the
            :py:mod:`app.hive_simulation` with -t flag, i.e., you are not using
            the multithreaded mode to speed up simulations.
    """

    #: A re-entrant lock used to make `eng` shareable by multiple threads.
    _LOCK = threading.RLock()
    #: A reference to the instance of `MatlabEngineContainer` or `None`.
    _instance: MatlabEngineContainer = None

    @staticmethod
    def get_instance() -> MatlabEngineContainer:
        """Used to obtain a singleton instance of :py:class:`MatlabEngineContainer`

        If one instance already exists that instance is returned,
        otherwise a new one is created and returned.

        Returns:
            A reference to the existing MatlabEngineContainer instance.
        """
        if MatlabEngineContainer._instance is None:
            with MatlabEngineContainer._LOCK:
                if MatlabEngineContainer._instance is None:
                    MatlabEngineContainer()
        return MatlabEngineContainer._instance

    def __init__(self) -> None:
        """Instantiates a new MatlabEngineContainer object.

        Note:
            Do not directly invoke constructor, instead use:
            :py:meth:`app.domain.helpers.matlab_utils.MatlabEngineContainer.getInstance`.
        """
        if MatlabEngineContainer._instance is None:
            print("Loading matlab engine... this can take a while.")
            try:
                self.eng = matlab.engine.start_matlab()
                self.eng.cd(MATLAB_DIR)
            except matlab.engine.EngineError:
                self.eng = None
            MatlabEngineContainer._instance = self
        else:
            raise RuntimeError("MatlabEngineContainer is a Singleton. Use "
                               "MatlabEngineContainer.getInstance() to get a "
                               "reference to a MatlabEngineContainer object.")

    def matrix_global_opt(self, a: np.ndarray, v_: np.ndarray) -> Any:
        """Constructs an optimized transition matrix using the matlab engine.

        Constructs an optimized transition matrix using linear programming
        relaxations and convex envelope approximations for the specified steady
        state ``v``, this is done by invoke the matlabscript matrixGlobalOpt
        in the project folder name matlab.

        Note:
            This function can only be invoked if you have a valid matlab license.

        Args:
            a:
                A non-optimized symmetric adjency matrix.
            v_:
                A stochastic steady state distribution vector.

        Returns:
            Markov Matrix with ``v_`` as steady state distribution and the
            respective mixing rate or None.
        """
        with MatlabEngineContainer._LOCK:
            ma = matlab.double(a.tolist())
            mv_ = matlab.double(v_.tolist())
            return self.eng.matrixGlobalOpt(ma, mv_, nargout=1)
