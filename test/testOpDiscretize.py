
import unittest

import numpy as np
import vigra

from lazyflow.graph import Graph

from deeplearning.targets import OpDiscretize
from deeplearning.targets import OpClassFromOneHot


class TestOpDiscretize(unittest.TestCase):
    def setUp(self):
        pass

    def testWithoutValid(self):
        x = np.asarray([.15, .25, .68, .83, .01])
        x = vigra.taggedView(x, axistags='t')

        g = Graph()
        op = OpDiscretize(graph=g)

        op.Input.setValue(x)
        op.NumLevels.setValue(5)

        exp = [[True, False, False, False, False],
               [False, True, False, False, False],
               [False, False, False, True, False],
               [False, False, False, False, True],
               [True, False, False, False, False]]
        exp = np.asarray(exp, dtype=np.bool)

        out = op.Output[...].wait()
        np.testing.assert_array_equal(out, exp)


class TestOpClassFromOneHot(unittest.TestCase):
    def setUp(self):
        pass

    def testWithoutValid(self):
        x = [[True, False, False, False, False],
             [False, True, False, False, False],
             [False, False, False, True, False],
             [False, False, False, False, True],
             [True, False, False, False, False]]
        x = np.asarray(x, dtype=np.bool)

        g = Graph()
        op = OpClassFromOneHot(graph=g)

        op.Input.setValue(x)

        out = op.Output[...].wait()
        exp = np.asarray([0, 1, 3, 4, 0], dtype=np.int)
        np.testing.assert_array_equal(out, exp)
