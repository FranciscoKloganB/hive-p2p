import os.path
import matlab.engine

from globals.globals import MATLAB_DIR


class MatlabEngineContainer:
    """Singleton class wrapper containing thread safe access to a MatlabEngine.


    Attrs:
        eng:
            A matlab engine instance object used for matrix and vector
            optimization operations throughout the simulations.
            Do not access this directly. MatlabEngine objects are not
            officially thread safe, thus it is recommended that you utilize
            the wrapped function, unless you are not running the
            :py:mod:`hive_simulation` with -t flag, i.e., you are not using
            the multithreaded mode to speed up simulations.
    """

    __instance = None

    @staticmethod
    def getInstance():
        """Used to obtain a singleton instance of :py:class:`MatlabEngineContainer`

        If one instance already exists that instance is returned,
        otherwise a new one is created and returned.

        Returns:
            A reference to the existing MatlabEngineContainer instance.
        """
        if MatlabEngineContainer.__instance is None:
            MatlabEngineContainer()
        return MatlabEngineContainer.__instance

    def __init__(self) -> None:
        """Instantiates a new MatlabEngineContainer object."""
        if MatlabEngineContainer.__instance is None:
            print("Loading matlab engine... this can take a while.")
            self.eng = matlab.engine.start_matlab()
            self.eng.cd(MATLAB_DIR)
            MatlabEngineContainer.__instance = self
        else:
            raise RuntimeError("MatlabEngineContainer is a Singleton. Use "
                               "MatlabEngineContainer.getInstance() to get a "
                               "reference to it.")
