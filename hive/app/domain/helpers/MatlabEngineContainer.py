from matlab import engine as me

from domain.Hive import MATLAB_DIR


class MatlabEngineContainer:
    """Singleton class wrapper containing thread safe access to a MatlabEngine.


    Attrs:
        __eng:
            A matlab engine instance object used for matrix and vector
            optimization operations throughout the simulations.
            Do not access this directly. MatlabEngine objects are not
            officially thread safe, thus it is recommended that you utilize
            the wrapped function.
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
            MatlabEngineContainer.__instance = self
            print("Loading matlab engine; This may take a few seconds...")
            self.__eng = me.start_matlab()
            self.__eng.cd(MATLAB_DIR)
            print("Matlab engine started. Resuming simulation...;")
            return
        raise RuntimeError("MatlabEngineContainer is a Singleton. Use "
                           "MatlabEngineContainer.getInstance() to get a "
                           "reference to it.")
