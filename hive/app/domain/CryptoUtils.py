import hashlib


def sha256(data, encoded=True):
    return hashlib.sha256(data).hexdigest() if encoded else hashlib.sha256(bytes(data, encoding='utf-8')).hexdigest()
