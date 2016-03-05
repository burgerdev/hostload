
import gc
from inspect import isframe
from pprint import pprint

from pympler import muppy

import numpy as np


def get_arrays():
    objs = muppy.get_objects()
    arrays = muppy.filter(objs, Type=np.ndarray)

    return arrays


def traverse(obj, ignore_list=None, start=False):
    if ignore_list is None:
        ignore_list = list()

    cancelled = False
    if not start:
        refs = gc.get_referrers(obj)
        refs = [r for r in refs if r not in ignore_list and not isframe(r)]
    else:
        refs = obj
    n = len(refs)

    while not cancelled:
        print(obj)
        print("Referrers:")
        pprint(refs)
        ans = raw_input("What now? [0-{}] or just ENTER\n".format(n-1))
        if len(ans) == 0:
            cancelled = True
        else:
            k = int(ans)
            traverse(refs[k], ignore_list=ignore_list + [refs], start=False)
