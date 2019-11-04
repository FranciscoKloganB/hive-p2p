import base64
import jsonpickle


def bytes_to_base64_string(data):
    if isinstance(data, str):
        return base64.b64encode(data.encode()).decode('utf-8')
    return base64.b64encode(data).decode('utf-8') if isinstance(data, bytes) else None


def base64_string_to_bytes(string):
    return base64.b64decode(string) if isinstance(string, str) else None


def base64_bytelike_obj_to_bytes(obj):
    return base64.b64decode(obj) if isinstance(obj, bytes) else None


def bytes_to_utf8string(data):
    return data.decode('utf-8') if isinstance(data, bytes) else None


def utf8string_to_bytes(string):
    return string.encode('utf-8') if isinstance(string, str) else None


def str_copy(string):
    return string.encode().decode()


def obj_to_json_string(obj):
    return jsonpickle.encode(obj)


def json_string_to_obj(json_string):
    return jsonpickle.decode(json_string)
