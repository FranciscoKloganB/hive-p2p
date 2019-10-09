from enum import Enum


class Status(Enum):
    SUSPECT = 1
    OFFLINE = 2
    ONLINE = 3


class HttpCodes(Enum):
    OK = 200
    NOT_FOUND = 404
    TIME_OUT = 408
    SERVER_DOWN = 521
