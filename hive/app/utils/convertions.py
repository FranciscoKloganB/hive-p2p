"""This module includes various data-type convertion utilities.

Some functions include representing a string or sequence of bytes
as base64-encoded strings or serialization objects into JSON strings.
"""

import base64
import importlib
import math

from typing import Union, Optional, Any, List

import jsonpickle


def bytes_to_base64_string(data: Union[bytes, str]) -> Optional[str]:
    """Converts a byte sequence or non base64 string to a base64-encoded string.

    Args:
        data: sequence of bytes to be converted.

    Returns:
        A base64-encoded string representation of data or None if data is not a
        str or bytes type.
    """
    if isinstance(data, str):
        return base64.b64encode(data.encode()).decode('utf-8')
    return base64.b64encode(data).decode('utf-8') if isinstance(data, bytes) else None


def base64_string_to_bytes(string: str) -> Optional[bytes]:
    """ Converts a base64 string to a sequence of bytes.

    Args:
        string: a base64-encoded string to be converted to a byte sequence.

    Returns:
        A sequence of bytes converted from the given base64-encoded string or
        None string is not a str type.
    """
    return base64.b64decode(string) if isinstance(string, str) else None


def base64_bytelike_obj_to_bytes(obj: bytes) -> Optional[bytes]:
    """Converts a byte-like object to a sequence of bytes.

    Args:
        obj: The object to be converted to base64-encoded string.

    Returns:
        A sequence of bytesrepresentation of the given object or None if obj
        is not a bytes type.
    """
    return base64.b64decode(obj) if isinstance(obj, bytes) else None


def bytes_to_utf8string(data: bytes) -> Optional[str]:
    """Converts a sequence of bytes a utf-8 string.

    Args:
        data: sequence of bytes to be converted.

    Returns:
        A utf-8 string representation of data or None if data is not a bytes
        type.
    """
    return data.decode('utf-8') if isinstance(data, bytes) else None


def utf8string_to_bytes(string: str) -> Optional[bytes]:
    """Converts utf-8 string to a sequence of bytes.

    Args:
        string: a utf-8 string to be converted to bytes.

    Returns:
        The bytes of the utf-8 string or None if string is not a str type.
    """
    return string.encode('utf-8') if isinstance(string, str) else None


def str_copy(string: str) -> Optional[str]:
    """Hard copies a string

     Note:
         Python's builtin copy.deepcopy() does not deep copy strings.

    Args:
        string: The string to be copied.

    Returns:
        An deep copy of the string or None if the string is not a str type.

    """
    return string.encode().decode() if isinstance(string, str) else None


def obj_to_json_string(obj: Any) -> str:
    """Serializes a python object to a JSON string.

    Args:
        obj: The object to be serialized.

    Returns:
        A string representation of the object in JSON format.
    """
    return jsonpickle.encode(obj)


def json_string_to_obj(json_string: str) -> Any:
    """Deserializes a JSON string to a a python object.

    Args:
        json_string: The string to be deserialized into a python object.

    Returns:
        A python object obtained from the JSON string.
    """
    return jsonpickle.decode(json_string)


def class_name_to_obj(module_name: str, class_name: str, args: List[Any]) -> Any:
    """Uses reflection to instanciate a class by name.

        Examples:
            The next two code snippets are equivalent:

                >>> class_name_to_obj(MASTER_SERVERS, "Master", ["f.jpg", 1, 80])

                >>> import app.domain.master_servers as ms
                >>> h = ms.Master("f.jpg", 1, 80)

    Args:
        module_name:
            The fully qualified path of the module the class is defined in.
            The name of the module must be included.
        class_name:
            The name of the class to be instanciated.
        args:
            The arguments expected by the named class as an iterable list.

    Returns:
        An object of the named class.

    Raises:
        AttributeError:
            When `class_name` does not exist or when `module_name` to be
            imported causes cyclic import errors.
        ImportError:
            When `module_name` is not a valid module.
    """
    module_ = importlib.import_module(module_name)
    instance = getattr(module_, class_name)(*args)
    return instance


def truncate_float_value(f: float, d: int) -> float:
    """Truncates a float value without rounding.

    Args:
        f:
            The float value to truncate.
        d:
            The number of decimal places the float can have.

    Returns:
        The truncated float.
    """
    return math.floor(f * 10 ** d) / 10 ** d
