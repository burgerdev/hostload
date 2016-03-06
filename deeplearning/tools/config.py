"""
Functions for dealing with config dictionaries for batch processing.

TODO describe expansion
"""

from itertools import product
from itertools import chain


def expand_dict(dict_, listify=True):
    """
    create several config dicts from one, expanding the lists inside
    """
    if listify:
        dict_ = listify_dict(dict_)

    if not isinstance(dict_, dict):
        return [dict_]

    keys = dict_.keys()
    lists = []
    for key in keys:
        expanded_subs = [expand_dict(item, listify=False)
                         for item in dict_[key]]
        lists.append(list(chain(*expanded_subs)))

    return [dict(zip(keys, combi)) for combi in product(*lists)]


def listify_dict(dict_):
    """
    turn each element that is not a list into a one-element list
    """
    if not isinstance(dict_, dict):
        return dict_

    def listify_sub(key):
        """
        actual worker
        """
        value = dict_[key]
        if isinstance(value, dict):
            return [listify_dict(value)]
        elif isinstance(value, list):
            return [listify_dict(el) for el in value]
        else:
            return [value]

    keys = dict_.keys()
    values = [listify_sub(key) for key in keys]
    return dict(zip(keys, values))
