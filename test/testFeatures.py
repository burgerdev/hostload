
import unittest

import numpy as np
import vigra

from lazyflow.graph import Graph

from tsdl.features import OpRawWindowed
from tsdl.features import OpDiff
from tsdl.features import OpMean
from tsdl.features import OpLinearWeightedMean
from tsdl.features import OpExponentialFilter
from tsdl.features import OpFairness
from tsdl.features import OpRecent
from tsdl.features import OpGaussianSmoothing


class TestOpMean(unittest.TestCase):
    def setUp(self):
        self.window_size = 3
        x = np.asarray([5, 7, 3, 4, 10, 2, 3])
        x = vigra.taggedView(x, axistags='t')
        self.data = x

    def testSimple(self):

        op, exp = self.getOp()
        op.Input.setValue(self.data)

        y = op.Output[...].wait()
        np.testing.assert_array_equal(y.shape, (7, 1))

        np.testing.assert_array_almost_equal(y.squeeze(), exp)

        y = op.Output[1:4].wait()
        np.testing.assert_array_equal(y.shape, (3, 1))

        np.testing.assert_array_almost_equal(y.squeeze(), exp[1:4])

    def testValid(self):

        op, exp = self.getOp()
        window = op.WindowSize.value
        a = window - 1
        b = 7 - a
        exp = np.asarray([0]*a + [1]*b)
        op.Input.setValue(self.data)

        y = op.Valid[...].wait()
        np.testing.assert_array_equal(y.shape, (7,))

        np.testing.assert_array_almost_equal(y, exp)

        y = op.Valid[1:4].wait()
        np.testing.assert_array_equal(y.shape, (3,))

        np.testing.assert_array_almost_equal(y, exp[1:4])

    def getOp(self):
        op = OpMean(graph=Graph())
        op.WindowSize.setValue(self.window_size)
        exp = np.asarray([5, 12, 15, 14, 17, 16, 15])/3.0
        return op, exp


class TestOpLinearWeightedMean(TestOpMean):
    def getOp(self):
        op = OpLinearWeightedMean(graph=Graph())
        op.WindowSize.setValue(self.window_size)
        exp = np.asarray([15, 31, 28, 25, 41, 30, 23])/6.0
        return op, exp


class TestOpExponentialFilter(TestOpMean):
    def getOp(self):
        op = OpExponentialFilter.build({"window_size": self.window_size},
                                       graph=Graph())
        exp = np.asarray([3.962407, 6.401044, 3.756507, 3.939616, 8.718104,
                          3.439447, 3.086751])
        return op, exp


class TestOpFairness(TestOpMean):
    def getOp(self):
        op = OpFairness(graph=Graph())
        op.WindowSize.setValue(self.window_size)
        exp = np.zeros((7,))
        exp[0] = 1.0
        exp[1] = (144)/float(25+49)
        exp[2] = (225)/float(25+49+9)
        exp[3] = (196)/float(49+9+16)
        exp[4] = (289)/float(9+16+100)
        exp[5] = (256)/float(16+100+4)
        exp[6] = (225)/float(100+4+9)
        exp = exp/3.0
        return op, exp


class TestOpRawWindowed(TestOpMean):
    def getOp(self):
        op = OpRawWindowed(graph=Graph())
        op.WindowSize.setValue(self.window_size)
        exp = np.asarray([5, 7, 3, 4, 10, 2, 3])
        return op, exp


class TestOpDiff(TestOpMean):
    def getOp(self):
        op = OpDiff.build(dict(), graph=Graph())
        op.WindowSize.setValue(2)
        exp = np.asarray([5, 2, -4, 1, 6, -8, 1])
        return op, exp


class TestOpRecent(TestOpMean):
    def testSimple(self):

        op, exp = self.getOp()
        op.Input.setValue(self.data)

        y = op.Output[...].wait()
        np.testing.assert_array_equal(y.shape, (7, 3))

        np.testing.assert_array_almost_equal(y, exp)

        y = op.Output[1:4, 0:1].wait()
        np.testing.assert_array_equal(y.shape, (3, 1))

        np.testing.assert_array_almost_equal(y, exp[1:4, 0:1])

    def getOp(self):
        d = {"class": OpRecent, "window_size": self.window_size}
        op = OpRecent.build(d, graph=Graph())
        exp = np.asarray([[5, 7, 3, 4, 10, 2, 3],
                          [5, 5, 7, 3, 4, 10, 2],
                          [5, 5, 5, 7, 3, 4, 10]]).T
        return op, exp


class TestOpGaussianSmoothing(unittest.TestCase):
    def setUp(self):
        self.window_size = 3
        x = np.asarray([5, 7, 3, 4, 10, 2, 3])
        x = vigra.taggedView(x, axistags='t')
        self.data = x

    def testSimple(self):

        op = OpGaussianSmoothing.build({"window_size": self.window_size},
                                       graph=Graph())
        op.Input.setValue(self.data)

        y = op.Output[...].wait()
