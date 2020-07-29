import hashlib
from typing import Any


def sha256(data: Any) -> str:
    """Calculates the SHA-256 hash of data. Data can be anything.

    Args:
        data: The data to get the hash from. If data is not of type bytes,
        it will be converted to bytes before the data is digested.

    Returns:
        The SHA-256 hash of data.

    """
    if isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    else:
        return hashlib.sha256(bytes(data, encoding='utf-8')).hexdigest()
