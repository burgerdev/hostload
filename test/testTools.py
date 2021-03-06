
import unittest
from pprint import pprint

import numpy as np
import vigra

from lazyflow.graph import Graph
from lazyflow.operators import Operator
from lazyflow.operators import OpArrayPiper

from tsdl.tools import OpArrayPiper as OpBuildableArrayPiper

from tsdl.tools import listify_dict
from tsdl.tools import expand_dict
from tsdl.tools import build_operator

from tsdl.tools.serialization import dumps
from tsdl.tools.serialization import loads

from tsdl.tools.generic import OpNormalize
from tsdl.tools.generic import OpChangeDtype


class TestTools(unittest.TestCase):
    def setUp(self):
        pass

    def testlistify_dict(self):
        d = {'a': [1, 2], 'b': 'x', 'c': {'d': 7, 'e': [Exception]},
             'f': (0, 1)}
        e = {'a': [1, 2], 'b': ['x'], 'c': [{'d': [7],
                                             'e': [Exception]}], 'f': [(0, 1)]}
        d2 = listify_dict(d)
        if e != d2:
            pprint(d)
            pprint(d2)
            raise AssertionError("not listified correctly")

    def testexpand_dict(self):
        d = {'a': [1, 2], 'b': ['x'],
             'c': [{'d': [7, 8], 'e': [Exception]}]}

        l = []
        l.append({'a': 1, 'b': 'x', 'c': {'d': 7, 'e': Exception}})
        l.append({'a': 1, 'b': 'x', 'c': {'d': 8, 'e': Exception}})
        l.append({'a': 2, 'b': 'x', 'c': {'d': 7, 'e': Exception}})
        l.append({'a': 2, 'b': 'x', 'c': {'d': 8, 'e': Exception}})

        l2 = list(expand_dict(d))

        if not contentEqual(l, l2):
            pprint(l)
            pprint(l2)
            raise AssertionError("expand_dict produced unexpected dicts")

    def testexpand_dictFull(self):
        d1 = {0: 0}
        d2 = {1: [1, 1.1]}
        d = {'a': (0, 1), 'b': [d1, d2]}

        l = []
        l.append({'a': (0, 1), 'b': d1})
        l.append({'a': (0, 1), 'b': {1: 1}})
        l.append({'a': (0, 1), 'b': {1: 1.1}})

        l2 = list(expand_dict(listify_dict(d)))

        if not contentEqual(l, l2):
            pprint(l)
            pprint(l2)
            raise AssertionError("expand_dict produced unexpected dicts")

    def testbuild_operator(self):
        class NotBuildable(OpArrayPiper):
            @classmethod
            def build(cls, config, parent=None, graph=None, workingdir=None):
                return cls(parent=parent, graph=graph)

        configs = ({"class": OpBuildableArrayPiper},
                   {"class": NotBuildable},
                   OpArrayPiper)
        kws = ({"graph": Graph()}, {"graph": Graph(), "workingdir": "temp"})

        for config in configs:
            for kwargs in kws:
                op = build_operator(config, **kwargs)
                print(op.__class__)
                assert isinstance(op, Operator), str(op)

    def testSerialization(self):
        from lazyflow.operator import Operator
        from tsdl.data import OpPickleCache
        d = {"class": Operator,
             "cache": {"class": OpPickleCache},
             "answer": 42,
             "a_list": [OpPickleCache, {"class": OpPickleCache,
                                        "tuple_in_dict": (1, 2, 3)}],
             "a_tuple": (OpPickleCache, {"class": OpPickleCache}),
             "nested": (1, [2, (3, 4, 5)]),
             "subdict": {"a": 1},
             "a_string": "asdf"}
        s = dumps(d)
        from pprint import pprint
        pprint(d)
        print("serialized to: \n{}".format(s))
        d2 = loads(s)
        print("")
        pprint(d2)
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

        op = OpNormalize.build({"mean": x.mean(), "stddev": np.sqrt(x.var())},
                               graph=Graph())
        op.Input.setValue(x)

        y = op.Output[...].wait()
        np.testing.assert_almost_equal(y.mean(), 0)
        np.testing.assert_almost_equal(y.var(), 1)

    def testChangeDtype(self):
        x = np.random.randint(0, 255, size=(1000,)).astype(np.int)
        x = vigra.taggedView(x, axistags='t')

        op = OpChangeDtype.build({}, graph=Graph())
        op.Input.setValue(x)

        y = op.Output[...].wait()
        assert y.dtype == np.float32

        op.Dtype.setValue(np.int)

        y = op.Output[...].wait()
        assert y.dtype == np.int


def contentEqual(a, b):
    return all(i in b for i in a) and all(i in a for i in b)
