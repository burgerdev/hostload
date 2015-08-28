
import unittest

import numpy as np
import vigra

from lazyflow.graph import Graph

from deeplearning.split import OpTrainTestSplit


class TestOpTrainTestSplit(unittest.TestCase):
    def setUp(self):
        pass

    def testWithoutValid(self):
        x = np.indices((100, 5)).sum(axis=0)
        x = vigra.taggedView(x, axistags='tc')

        g = Graph()
        op = OpTrainTestSplit(graph=g)

        op.Input.setValue(x)
        op.TestPercentage.setValue(.1)
        op.ValidPercentage.setValue(0)

        assert op.Train.meta.shape == (90, 5)
        assert op.Test.meta.shape == (10, 5)

        y = np.concatenate((op.Train[...].wait(), op.Test[...].wait()), axis=0)
        np.testing.assert_array_equal(x.view(np.ndarray), y)

    def testWithValid(self):
        x = np.indices((100, 5)).sum(axis=0)
        x = vigra.taggedView(x, axistags='tc')

        g = Graph()
        op = OpTrainTestSplit(graph=g)

        op.Input.setValue(x)
        op.TestPercentage.setValue(.2)
        op.ValidPercentage.setValue(.1)

        np.testing.assert_array_equal(op.Train.meta.shape, (72, 5))
        np.testing.assert_array_equal(op.Test.meta.shape, (20, 5))
        np.testing.assert_array_equal(op.Valid.meta.shape, (8, 5))

        d = op.Description[...].wait()
        np.testing.assert_array_equal(d[:72, :], np.zeros((72, 5)))
        np.testing.assert_array_equal(d[72:80, :], np.ones((8, 5)))
        np.testing.assert_array_equal(d[80:, :], 2*np.ones((20, 5)))

        d = op.Description[71:73, 0:1].wait().squeeze()
        expected = np.asarray([0, 1], dtype=np.int)
        np.testing.assert_array_equal(d, expected)
