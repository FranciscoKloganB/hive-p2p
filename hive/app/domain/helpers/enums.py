from __future__ import annotations

from enum import Enum


class Status(Enum):
    """Enumerator that defines connectivity status of a network node

    The following status exist:
        * SUSPECT: Network node may be offline;
        * OFFLINE: Network node is offline;
        * ONLINE: Network node is online;
    """
    SUSPECT: int = 1
    OFFLINE: int = 2
    ONLINE: int = 3


class HttpCodes(Enum):
    """Enumerator class used to represent HTTP response codes

    The following codes are considered:
        * DUMMY: Dummy value. Use when no valid HTTP code exists;
        * OK: Callee accepts the sent file;
        * BAD_REQUEST: Callee refuses the integrity of sent file;
        * NOT_FOUND: Callee is not a member of the network;
        * NOT_ACCEPTABLE: Callee already has a file with same Id;
        * TIME_OUT: Message lost in translation;
        *  SERVER_DOWN: Metadata server is offline;
    """
    DUMMY: int = -1
    OK: int = 200
    BAD_REQUEST: int = 400
    NOT_FOUND: int = 404
    NOT_ACCEPTABLE: int = 406
    TIME_OUT: int = 408
    SERVER_DOWN: int = 521

