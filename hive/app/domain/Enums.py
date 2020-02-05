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
    OK: int = 200
    BAD_REQUEST: int = 400
    NOT_FOUND: int = 404
    NOT_ACCEPTABLE: int = 406
    TIME_OUT: int = 408
    SERVER_DOWN: int = 521
