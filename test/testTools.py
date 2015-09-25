
import unittest
from pprint import pprint

import numpy as np
import vigra

from lazyflow.graph import Graph

from deeplearning.tools import listifyDict
from deeplearning.tools import expandDict

from deeplearning.tools.serialization import dumps
from deeplearning.tools.serialization import loads

from deeplearning.tools.generic import OpNormalize


class TestTools(unittest.TestCase):
    def setUp(self):
        pass

    def testListifyDict(self):
        d = {'a': [1, 2], 'b': 'x', 'c': {'d': 7, 'e': [Exception]}}
        e = {'a': [1, 2], 'b': ['x'], 'c': [{'d': [7],
                                             'e': [Exception]}]}
        d2 = listifyDict(d)
        if e != d2:
            pprint(d)
            pprint(d2)
            raise AssertionError("not listified correctly")

    def testExpandDict(self):
        d = {'a': [1, 2], 'b': ['x'],
             'c': [{'d': [7, 8], 'e': [Exception]}]}

        l = []
        l.append({'a': 1, 'b': 'x', 'c': {'d': 7, 'e': Exception}})
        l.append({'a': 1, 'b': 'x', 'c': {'d': 8, 'e': Exception}})
        l.append({'a': 2, 'b': 'x', 'c': {'d': 7, 'e': Exception}})
        l.append({'a': 2, 'b': 'x', 'c': {'d': 8, 'e': Exception}})

        l2 = list(expandDict(d))

        if not contentEqual(l, l2):
            pprint(l)
            pprint(l2)
            raise AssertionError("expandDict produced unexpected dicts")

    def testSerialization(self):
        from lazyflow.operator import Operator
        from deeplearning.data import OpPickleCache
        d = {"class": Operator,
             "cache": {"class": OpPickleCache},
             "answer": 42,
             "subdict": {"a": 1}}
        s = dumps(d)
        print("serialized to: \n{}".format(s))
        d2 = loads(s)
        print("")
        print(d)
        print(d2)
        assert d == d2

        class Custom(object):
            A = 1

        d = {"key": Custom()}
        with self.assertRaises(TypeError):
            dumps(d)

    def testNormalize(self):
        x = np.random.random(size=(1000,))
        x = vigra.taggedView(x, axistags='t')

        op = OpNormalize(graph=Graph())
        op.Input.setValue(x)

        y = op.Output[...].wait()
        np.testing.assert_almost_equal(y.mean(), x.mean())
        np.testing.assert_almost_equal(y.var(), x.var())

        op.Mean.setValue(x.mean())
        op.StdDev.setValue(np.sqrt(x.var()))

        y = op.Output[...].wait()
        np.testing.assert_almost_equal(y.mean(), 0)
        np.testing.assert_almost_equal(y.var(), 1)


def contentEqual(a, b):
    return all(i in b for i in a) and all(i in a for i in b)
