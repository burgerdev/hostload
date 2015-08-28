
import unittest

import numpy as np
import vigra

from lazyflow.graph import Graph

from deeplearning.targets import OpExponentiallySegmentedPattern


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
