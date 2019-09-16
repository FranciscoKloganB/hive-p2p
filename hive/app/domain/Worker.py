import psutil

from pathlib import Path
from domain.SharedFilePart import SharedFilePart


class Worker:
    """
    Defines a node on the P2P network. Workers are subject to constraints imposed by Hiveminds, constraints they inflict
    on themselves based on available computing power (CPU, RAM, etc...) and can have [0, N] shared file parts. Workers
    have the ability to reconstruct lost file parts whene needed.
    :ivar shared_file_parts: part_name is a key to a dict of integer part_id keys leading to actual SharedFileParts
    :type dict<string, dict<int, SharedFilePart>>
    """

    def __init__(self):
        self.shared_file_parts = {}
