from __future__ import annotations

import json

import numpy as np
import pandas as pd
import domain.Hive as h

from pathlib import Path
from tabulate import tabulate
from typing import Any, Union, Dict

from globals.globals import *
from domain.helpers.SimulationData import SimulationData


class FileData:
    """
    Helper class for domain.Hivemind to keep track of how many parts exist of a file, the number of file parts expected
    to be within the long-term highest density node among other information.
    :ivar str name: the name of the original file
    :ivar int parts_count: the total number of parts the original file was split into, excluding replicas
    :ivar pd.DataFrame desired_distribution: file density distribution hive members must achieve with independent realizations for ideal persistence of the file
    :ivar pd.DataFrame current_distribution: tracks current of file distribution, updated at each epoch
    :ivar ConvergenceData simulation_data: instance object with general information w.r.t. the simulation
    """

    # region Class Variables, Instance Variables and Constructors
    def __init__(self, name: str, sim_number: int = 0, origin: str = ""):
        """
        :param str name: name of the file referenced by this data class instance
        :param int sim_number: optional value that can be passed to FileData to generate different .out names
        """
        self.name: str = name
        self.desired_distribution: Union[None, pd.DataFrame] = None
        self.current_distribution: Union[None, pd.DataFrame] = None
        self.simulation_data: SimulationData = SimulationData()
        self.out_file: Any = open(
            os.path.join(
                OUTFILE_ROOT, "{}_{}{}".format(
                    Path(name).resolve().stem,
                    Path(origin).resolve().stem,
                    sim_number,
                    ".out" if DEBUG else ".json"
                )
            ), "w+")
    # endregion

    # region Instance Methods
    def equal_distributions(self, parts_in_hive: int) -> bool:
        """
        Delegates distribution comparison to ConvergenceData.equal_distributions static method
        """
        if parts_in_hive == 0:
            return False

        normalized_cdv = self.current_distribution.divide(parts_in_hive)
        if DEBUG:
            self.fwrite("Desired Distribution:\n{}\nCurrent Distribution:\n{}\n".format(
                tabulate(self.desired_distribution, headers='keys', tablefmt='psql'), tabulate(normalized_cdv, headers='keys', tablefmt='psql')
            ))
        return np.allclose(self.desired_distribution, normalized_cdv, rtol=R_TOL, atol=(1 / parts_in_hive))
    # endregion

    # region File I/O
    def fwrite(self, string: str) -> None:
        """
        Writes to the out_file referenced by the FileData's instance out_file field in append mode
        :param str string: a message to write to file
        """
        self.out_file.write(string + "\n")

    def jwrite(self, hive: h.Hive, origin: str, epoch: int):
        sim_data: SimulationData = self.simulation_data
        if not sim_data.msg:
            sim_data.msg.append("completed simulation successfully")
        if DEBUG:
            [print("* {};".format(reason)) for reason in sim_data.msg]

        sim_data.corrupted_parts = sim_data.corrupted_parts[:epoch]
        sim_data.delay = sim_data.delay[:epoch]
        sim_data.disconnected_workers = sim_data.disconnected_workers[:epoch]
        sim_data.lost_messages = sim_data.lost_messages[:epoch]
        sim_data.lost_parts = sim_data.lost_parts[:epoch]
        sim_data.hive_status_before_maintenance = sim_data.hive_status_before_maintenance[:epoch]
        sim_data.hive_size_before_maintenance = sim_data.hive_size_before_maintenance[:epoch]
        sim_data.hive_size_after_maintenance = sim_data.hive_size_after_maintenance[:epoch]
        sim_data.moved_parts = sim_data.moved_parts[:epoch]

        extras: Dict[str, Any] = {
            "simfile_name": origin,
            "hive_id": hive.id,
            "file_name": self.name,
            "read_size": READ_SIZE,
            "critical_size_threshold": hive.critical_size,
            "sufficient_size_threshold": hive.sufficient_size,
            "original_hive_size": hive.original_size,
            "redundant_size": hive.redundant_size,
            "max_epochs": MAX_EPOCHS,
            "min_recovery_delay": MIN_DETECTION_DELAY,
            "max_recovery_delay": MAX_DETECTION_DELAY,
            "replication_level": REPLICATION_LEVEL,
            "convergence_treshold": MIN_CONVERGENCE_THRESHOLD,
            "channel_loss": LOSS_CHANCE,
            "corruption_chance_tod": hive.corruption_chances[0]
        }

        sim_data_dict = sim_data.__dict__
        sim_data_dict.update(extras)
        json_string = json.dumps(sim_data_dict, indent=4, sort_keys=True, ensure_ascii=False)
        self.fwrite(json_string)

    def fclose(self, string: str = None) -> None:
        """
        Closes the out_file referenced by the FileData's instance out_file field
        :param str string: if filled, a message is written in append mode before closing the out_file
        """
        if string:
            self.fwrite(string)
        self.out_file.close()
    # endregion

    # region Overrides
    def __hash__(self):
        # allows a worker object to be used as a dictionary key
        return hash(str(self.name))

    def __eq__(self, other):
        if not isinstance(other, FileData):
            return False
        return self.name == other.name

    def __ne__(self, other):
        return not(self == other)
    # endregion

    # region Helpers
    def reset_convergence_data(self) -> None:
        """
        Resets the FileData instance field simulation_data by delegation to ConvergenceData instance method
        """
        self.simulation_data.save_sets_and_reset()
    # endregion

