
from itertools import product
from itertools import chain

import numpy as np


def expandDict(d, listify=True):
    if listify:
        d = listifyDict(d)

    if not isinstance(d, dict):
        return [d]

    keys = d.keys()
    lists = []
    for k in keys:
        l = [expandDict(item, listify=False) for item in d[k]]
        lists.append(list(chain(*l)))

    return [dict(zip(keys, combi)) for combi in product(*lists)]


def listifyDict(d):
    if not isinstance(d, dict):
        return d

    def listify_sub(k):
        v = d[k]
        if isinstance(v, dict):
            return [listifyDict(v)]
        elif isinstance(v, list):
            return [listifyDict(el) for el in v]
        else:
            return [v]

    keys = d.keys()
    values = [listify_sub(k) for k in keys]
    return dict(zip(keys, values))


def get_rng():
    return np.random.RandomState(420)
