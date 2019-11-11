from enum import Enum


class Status(Enum):
    SUSPECT: int = 1
    OFFLINE: int = 2
    ONLINE: int = 3


class HttpCodes(Enum):
    OK: int = 200
    NOT_FOUND: int = 404
    TIME_OUT: int = 408
    SERVER_DOWN: int = 521
