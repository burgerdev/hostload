
import json
import re
from importlib import import_module


_class_re = re.compile("^<class '(.*)'>$")


def _encode(obj):
    if isinstance(obj, list) or isinstance(obj, tuple):
        return _encode_listlike(obj)
    elif isinstance(obj, dict):
        return _encode_dict(obj)
    elif isinstance(obj, str) or isinstance(obj, unicode):
        return _encode_string(obj)
    elif isinstance(obj, type):
        return str(obj)
    else:
        return obj


def _encode_string(string):
    match = _class_re.match(string)
    if match is None:
        return string
    else:
        class_string = match.group(1)
        itemized = class_string.split(".")
        class_name = itemized[-1]
        module = import_module(".".join(itemized[:-1]))
        return getattr(module, class_name)


def _encode_listlike(value):
    list_ = [_encode(v) for v in value]
    if isinstance(value, tuple):
        value = tuple(list_) + ("t",)
    elif isinstance(value, list):
        value = list_ + ["l"]
    return value


def _encode_dict(dict_from):
    dict_to = dict()

    for key in dict_from:
        value = dict_from[key]
        value = _encode(value)
        dict_to[key] = value
    return dict_to


def _decode(obj):
    if isinstance(obj, list) or isinstance(obj, tuple):
        return _decode_listlike(obj)
    elif isinstance(obj, dict):
        return _decode_dict(obj)
    elif isinstance(obj, str) or isinstance(obj, unicode):
        return _decode_string(obj)
    else:
        return obj


def _decode_string(string):
    match = _class_re.match(string)
    if match is None:
        return string
    else:
        class_string = match.group(1)
        itemized = class_string.split(".")
        class_name = itemized[-1]
        module = import_module(".".join(itemized[:-1]))
        return getattr(module, class_name)


def _decode_listlike(value):
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
    dict_to = dict()

    for key in dict_from:
        value = dict_from[key]
        value = _decode(value)
        dict_to[key] = value
    return dict_to


def dumps(obj, **kwargs):
    obj = _encode(obj)
    return json.dumps(obj, **kwargs)


def loads(string, **kwargs):
    dict_from = json.loads(string, **kwargs)
    return _decode(dict_from)


#loads = partial(json.loads, object_hook=_CustomDecoder.decoding_hook)
