import hashlib


def sha256(data):
    if isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    else:
        return hashlib.sha256(bytes(data, encoding='utf-8')).hexdigest()
