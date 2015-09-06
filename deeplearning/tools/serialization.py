
import json
import re
from importlib import import_module

from functools import partial


class _CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, type):
            return str(obj)
        return super(_CustomEncoder, self).default(obj)


_class_re = re.compile("^<class '(.*)'>$")

def _decodingHook(dct):
    if "class" not in dct:
        return dct
    m = _class_re.match(dct["class"])
    if m is not None:
        class_string = m.group(1)
        itemized = class_string.split(".")
        class_name = itemized[-1]
        module = import_module(".".join(itemized[:-1]))
        dct["class"] = getattr(module, class_name)
    return dct

dumps = partial(json.dumps, cls=_CustomEncoder)
loads = partial(json.loads, object_hook=_decodingHook)
