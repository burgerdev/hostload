"""
provides methods for storing and loading configurations

Persisting configurations is not straight-forward because many serialization
packages are
  * not accessible for humans (pickle) or
  * oblivious to the difference between tuple and list (json, yaml)

We define a wrapper around the json module to handle this. Note that there
will be a special tag at the end of each json list indicating whether it was
a tuple or a list in the first place. While deserializing, we try to be as
tolerant as possible, in case a user entered a dict manually and forgot the
tag.
"""
import json
import re
from importlib import import_module


_CLASS_RE = re.compile("^<class '(.*)'>$")


def _encode(obj):
    """
    encode an object for use with json (used recursively)

    the dict can contain following values
      * types (classes)
      * int, float, str, unicode
      * list, tuple
      * further dicts (conforming to this list)

    Keys must be json serializable, and using strings as keys is strongly
    recommended.
    """
    if isinstance(obj, list) or isinstance(obj, tuple):
        return _encode_listlike(obj)
    elif isinstance(obj, dict):
        return _encode_dict(obj)
    elif isinstance(obj, type):
        return str(obj)
    else:
        return obj


def _encode_listlike(value):
    """
    handle list instance
    """
    list_ = [_encode(v) for v in value]
    if isinstance(value, tuple):
        value = tuple(list_) + ("t",)
    elif isinstance(value, list):
        value = list_ + ["l"]
    return value


def _encode_dict(dict_from):
    """
    encode each value in a dict
    """
    dict_to = dict()

    for key in dict_from:
        value = dict_from[key]
        value = _encode(value)
        dict_to[key] = value
    return dict_to


def _decode(obj):
    """
    revert the encoding process by _encode()
    """
    if isinstance(obj, list) or isinstance(obj, tuple):
        return _decode_listlike(obj)
    elif isinstance(obj, dict):
        return _decode_dict(obj)
    elif isinstance(obj, str) or isinstance(obj, unicode):
        return _decode_string(obj)
    else:
        return obj


def _decode_string(string):
    """
    disentangle real strings and types
    """
    match = _CLASS_RE.match(string)
    if match is None:
        return string
    else:
        class_string = match.group(1)
        itemized = class_string.split(".")
        class_name = itemized[-1]
        module = import_module(".".join(itemized[:-1]))
        return getattr(module, class_name)


def _decode_listlike(value):
    """
    disentangle lists and tuples
    """
    value = [_decode(v) for v in value]

    if len(value) == 0:
        pass
    elif not (isinstance(value[-1], str) or
              isinstance(value[-1], unicode)):
        pass
    elif value[-1] not in "tl":
        pass
    elif value[-1] == "t":
        value = tuple(value[:-1])
    elif value[-1] == "l":
        value = value[:-1]
    return value


def _decode_dict(dict_from):
    """
    decode all values in a dict
    """
    dict_to = dict()

    for key in dict_from:
        value = dict_from[key]
        value = _decode(value)
        dict_to[key] = value
    return dict_to


def dumps(obj, **kwargs):
    """
    convert configuration object to a json string
    """
    obj = _encode(obj)
    return json.dumps(obj, **kwargs)


def loads(string, **kwargs):
    """
    reconstruct configuration object from json string
    """
    dict_from = json.loads(string, **kwargs)
    return _decode(dict_from)
