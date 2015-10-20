
import json
import re
from importlib import import_module

from functools import partial


class _CustomEncoder(json.JSONEncoder):
    @classmethod
    def _handle_list_like(cls, value):
        if isinstance(value, tuple):
            value = tuple(map(cls._handle_list_like, value)) + ("t",)
        elif isinstance(value, list):
            value = map(cls._handle_list_like, value) + ["l"]
        return value

    def encode(self, obj):
        assert isinstance(obj, dict)
        obj = obj.copy()
        for key in obj:
            obj[key] = self._handle_list_like(obj[key])
        return super(_CustomEncoder, self).encode(obj)

    def default(self, obj):
        if isinstance(obj, type):
            return str(obj)
        return super(_CustomEncoder, self).default(obj)


class _CustomDecoder(object):
    _class_re = re.compile("^<class '(.*)'>$")

    @classmethod
    def _handle_list_like(cls, value):
        if isinstance(value, list):
            value = map(cls._handle_list_like, value)
            if len(value) == 0 or value[-1] not in "tl":
                pass
            elif value[-1] == "t":
                value = tuple(value[:-1])
            elif value[-1] == "l":
                value = value[:-1]
        else:
            value = cls.decode_class(value)
        return value

    @classmethod
    def decode_class(cls, string):
        if not (isinstance(string, str) or isinstance(string, unicode)):
            return string

        match = cls._class_re.match(string)
        if match is None:
            return string
        else:
            class_string = match.group(1)
            itemized = class_string.split(".")
            class_name = itemized[-1]
            module = import_module(".".join(itemized[:-1]))
            return getattr(module, class_name)

    @classmethod
    def decoding_hook(cls, dict_):
        for key in dict_:
            dict_[key] = cls._handle_list_like(dict_[key])

        if "class" in dict_:
            dict_["class"] = cls.decode_class(dict_["class"])

        return dict_

dumps = partial(json.dumps, cls=_CustomEncoder)
loads = partial(json.loads, object_hook=_CustomDecoder.decoding_hook)
