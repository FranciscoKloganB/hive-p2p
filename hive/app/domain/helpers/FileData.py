from domain.helpers.ConvergenceData import ConvergenceData

class FileData:
    # region docstrings
    """
    Helper class for domain.Hivemind to keep track of how many parts exist of a file, the number of file parts expected
    to be within the long-term highest density node among other information.
    :ivar file_name: the name of the file
    :type str
    :ivar parts_count: how many parts exist for the
    :type int
    :ivar highest_density_node: label of the highest density node
    :type str
    :ivar highest_density_node_density: file density for the highest density node
    :type float
    :ivar desired_distribution: desired distribution that sharers of this file must achieve together but independently
    :type pandas.Dataframe, column vector with labels
    :ivar current_distritubiton: keeps track of file current distribution, at each discrete time stage
    :type dict<str, list<float>>
    :ivar convergence_data: instance object with general information perteining the simulation
    :type domain.helpers.ConvergenceData
    """
    # endregion

    # region class variables, instance variables and constructors
    def __init__(self,
                 file_name="",
                 parts_count=0,
                 node_name="",
                 density=0.0,
                 ddv=None,
                 cdv=None,
                 convergence_data=None):
        self.file_name = file_name
        self.parts_count = parts_count
        self.highest_density_node = node_name
        self.highest_density_node_density = density
        self.desired_distribution = ddv
        self.current_distritubiton = cdv
        self.convergence_data = convergence_data
    # endregion
