
import unittest

import numpy as np
import vigra

from lazyflow.graph import Graph

from tsdl.split import OpTrainTestSplit


class TestOpTrainTestSplit(unittest.TestCase):
    def setUp(self):
        pass

    def testWithoutValid(self):
        x = np.indices((100, 5)).sum(axis=0)
        x = vigra.taggedView(x, axistags='tc')

        g = Graph()
        op = OpTrainTestSplit.build(dict(test=.1, valid=.0), graph=g)

        op.Input.resize(1)

        op.Input[0].setValue(x)

        assert op.Train[0].meta.shape == (90, 5)
        assert op.Test[0].meta.shape == (10, 5)

        y = np.concatenate((op.Train[0][...].wait(),
                            op.Test[0][...].wait()), axis=0)
        np.testing.assert_array_equal(x.view(np.ndarray), y)

    def testDescription(self):
        x = np.indices((100, 5)).sum(axis=0)
        x = vigra.taggedView(x, axistags='tc')

        g = Graph()
        op = OpTrainTestSplit(graph=g)

        op.Input.resize(1)

        op.Input[0].setValue(x)
        op.TestPercentage.setValue(.2)
        op.ValidPercentage.setValue(.1)

        np.testing.assert_array_equal(op.Train[0].meta.shape, (72, 5))
        np.testing.assert_array_equal(op.Test[0].meta.shape, (20, 5))
        np.testing.assert_array_equal(op.Valid[0].meta.shape, (8, 5))

        d = op.Description[...].wait()
        np.testing.assert_array_equal(d[:72], np.zeros((72,)))
        np.testing.assert_array_equal(d[72:80], np.ones((8,)))
        np.testing.assert_array_equal(d[80:], 2*np.ones((20,)))

        d = op.Description[71:73].wait()
        expected = np.asarray([0, 1], dtype=np.int)
        np.testing.assert_array_equal(d, expected)
