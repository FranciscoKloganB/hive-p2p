from __future__ import annotations

from enum import Enum


class Status(Enum):
    """
    Enumerator class used to represent if a given Worker or super node is online, offline among other possible status
    """
    SUSPECT: int = 1
    OFFLINE: int = 2
    ONLINE: int = 3


class HttpCodes(Enum):
    """
    Enumerator class used to represent HTTP response codes
    """
    DUMMY: int = -1
    OK: int = 200  # used when destination accepts the sent file
    BAD_REQUEST: int = 400  # used when destination considers the sent file incorrect integrity wise
    NOT_FOUND: int = 404  # used when destination is not in the hive
    NOT_ACCEPTABLE: int = 406  # used when destination already has the part that was sent by sender
    TIME_OUT: int = 408  # used when a message is lost in translation
    SERVER_DOWN: int = 521

