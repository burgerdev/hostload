
import unittest

import numpy as np
import vigra

from lazyflow.graph import Graph

from deeplearning.targets import OpExponentiallySegmentedPattern

from deeplearning.targets import OpDiscretize
from deeplearning.targets import OpClassFromOneHot

from deeplearning.targets import OpHostloadTarget


class TestOpExponentiallySegmentedpattern(unittest.TestCase):
    def setUp(self):
        self.baseline_size = 2
        self.num_segments = 2
        x = np.asarray([5, 7, 3, 4, 10, 2, 3])
        x = vigra.taggedView(x, axistags='t')
        self.data = x

    def testSimple(self):
        x = self.data
        op = OpExponentiallySegmentedPattern(graph=Graph())
        op.NumSegments.setValue(self.num_segments)
        op.BaselineSize.setValue(self.baseline_size)
        op.Input.setValue(x)
        exp = np.asarray([[6, 5, 3.5, 7, 6, 2.5, 1.5],
                          [4.75, 6, 4.75, 4.75, 3.75, 1.25, .75]]).T

        y = op.Output[...].wait()
        np.testing.assert_array_equal(y.shape, (7, 2))

        np.testing.assert_array_almost_equal(y.T, exp.T)

        y = op.Output[1:4, ...].wait()
        np.testing.assert_array_equal(y.shape, (3, 2))

        np.testing.assert_array_almost_equal(y.T, exp[1:4, :].T)


class TestOpDiscretize(unittest.TestCase):
    def setUp(self):
        pass

    def testSimple(self):
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
        exp = np.asarray(exp, dtype=np.bool).astype(np.float)

        out = op.Output[...].wait()
        np.testing.assert_array_equal(out, exp)


class TestOpClassFromOneHot(unittest.TestCase):
    def setUp(self):
        pass

    def testSimple(self):
        x = [[True, False, False, False, False],
             [False, True, False, False, False],
             [False, False, False, True, False],
             [False, False, False, False, True],
             [True, False, False, False, False]]
        x = np.asarray(x, dtype=np.bool).astype(np.float)

        g = Graph()
        op = OpClassFromOneHot(graph=g)

        op.Input.setValue(x)

        out = op.Output[...].wait()
        exp = np.asarray([0, 1, 3, 4, 0], dtype=np.int)
        np.testing.assert_array_equal(out, exp)


class TestOpHostloadTarget(unittest.TestCase):
    def setUp(self):
        self.window_size = 2
        self.num_levels = 3
        x = np.asarray([.5, .7, .3, .4, 1.0, .2, .3])
        x = vigra.taggedView(x, axistags='t')
        self.data = x

        expected = np.asarray([[0, 1, 0],
                               [0, 1, 0],
                               [0, 1, 0],
                               [0, 0, 1],
                               [0, 1, 0],
                               [1, 0, 0],
                               [1, 0, 0]])
        self.expected = expected

    def testSimple(self):
        op = OpHostloadTarget(graph=Graph())
        op.Input.setValue(self.data)
        op.WindowSize.setValue(self.window_size)
        op.NumLevels.setValue(self.num_levels)
        out = op.Output[...].wait()
        np.testing.assert_array_equal(out, self.expected)
