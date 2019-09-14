def string_to_utf8_bytes(string):
    return string.encode('utf-8') if isinstance(string, str) else None


def utf8_bytes_to_string(data):
    return data.decode('utf-8') if isinstance(data, bytes) else None


