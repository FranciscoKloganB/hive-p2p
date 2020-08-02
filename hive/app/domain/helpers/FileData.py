from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, IO

import domain.Hive as h
from domain.helpers.SimulationData import SimulationData
from globals.globals import *
from hive_simulation import MAX_EPOCHS


class FileData:
    """Holds essential simulation data concerning files being persisted.

    FileData is a helper class which has responsabilities such as tracking
    how many parts including replicas exist of the named file and managing
    the persistence of logged simulation data to disk.

    Attributes:
        name (str):
            The name of the original file.
        parts_in_hive (int):
            The number of file parts including replicas that exist for the
            named file that exist in the simulation. Updated every epoch.
        simulation_data (SimulationData):
            Object that stores captured simulation data. Stored data can be
            post-processed using user defined scripts to create items such
            has graphs and figures. See :py:class:`SimulationData
            <domain.helpers.SimulationData.SimulationData`
        out_file (str/bytes/int):
            File output stream to where captured data is written in append mode.
    """

    def __init__(self, name: str, sim_id: int = 0, origin: str = "") -> None:
        """Creates an instance of FileData

        Args:
            name:
                Name of the file to be referenced by the FileData object.
            sim_id:
                optional; Identifier that generates unique output file names,
                thus guaranteeing that different simulation instances do not
                overwrite previous out files.
            origin:
                optional; The name of the simulation file name that started
                the simulation process.
        """
        self.name: str = name
        self.parts_in_hive = 0
        self.simulation_data: SimulationData = SimulationData()
        self.out_file: IO = open(
            os.path.join(
                OUTFILE_ROOT, "{}_{}{}.{}".format(
                    Path(name).resolve().stem,
                    Path(origin).resolve().stem,
                    sim_id,
                    "json")
            ), "w+")

    def fwrite(self, msg: str) -> None:
        """Writes a message to the output file referenced by the FileData object.

        The method fwrite automatically adds a new line to the inputted message.

        Args:
            msg:
                The message to be logged on the output file.
        """
        self.out_file.write(msg + "\n")

    def jwrite(self, hive: h.Hive, origin: str, epoch: int) -> None:
        """Writes a JSON string of the SimulationData instance to the output file.

        The logged data is defined by the attributes of the
        :py:class:`SimulationData <domain.helpers.SimulationData.SimulationData`
         class.

        Args:
            hive:
                The :py:class:`Hive <domain.Hive.Hive>` object that manages
                the simulated persistence of the referenced file.
            origin:
                The name of the simulation file that started the simulation
                process.
            epoch:
                The epoch at which the SimulationData was logged into the
                output file.

        """
        sd: SimulationData = self.simulation_data

        sd.save_sets_and_reset()

        if not sd.messages:
            sd.messages.append("completed simulation successfully")

        sd.parts_in_hive = sd.parts_in_hive[:epoch]

        sd.disconnected_workers = sd.disconnected_workers[:epoch]
        sd.lost_parts = sd.lost_parts[:epoch]

        sd.hive_status_before_maintenance = sd.hive_status_before_maintenance[:epoch]
        sd.hive_size_before_maintenance = sd.hive_size_before_maintenance[:epoch]
        sd.hive_size_after_maintenance = sd.hive_size_after_maintenance[:epoch]

        sd.delay = sd.delay[:epoch]

        sd.moved_parts = sd.moved_parts[:epoch]
        sd.corrupted_parts = sd.corrupted_parts[:epoch]
        sd.lost_messages = sd.lost_messages[:epoch]

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

        sim_data_dict = sd.__dict__
        sim_data_dict.update(extras)
        json_string = json.dumps(
            sim_data_dict, indent=4, sort_keys=True, ensure_ascii=False)

        self.fwrite(json_string)

    def fclose(self, msg: str = None) -> None:
        """Closes the output file controlled by the FileData instance.

        Args:
             msg:
                optional; If filled, a termination message is logged into the
                output file that is being closed.
        """
        if msg:
            self.fwrite(msg)
        self.out_file.close()

    # region Overrides

    def __hash__(self):
        """Override to allows a network node object to be used as a dict key

        Returns:
            The hash of value of the referenced file :py:attr:`~name`.
        """
        return hash(str(self.name))

    def __eq__(self, other):
        """Compares if two instances of FileData are equal.

        Equality is based on name equality.

        Returns:
            True if the name attribute of both instances is the same,
            otherwise False.
        """
        if not isinstance(other, FileData):
            return False
        return self.name == other.name

    def __ne__(self, other):
        """Compares if two instances of FileData are not equal."""
        return not(self == other)

    # endregion
