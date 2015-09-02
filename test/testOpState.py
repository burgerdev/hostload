
import unittest

import numpy as np
import vigra

from lazyflow.graph import Graph

from deeplearning.classifiers import OpStateTrain
from deeplearning.classifiers import OpStatePredict


class TestOpState(unittest.TestCase):
    def setUp(self):
        X = np.array([[1.2, 5], [2.2, 8], [3.1, 15], [3.9, 17]])
        X = vigra.taggedView(X, axistags='tc')
        y = np.array([1, 2, 3, 4], dtype=np.int)
        y = vigra.taggedView(y, axistags='t')

        self.X = X
        self.y = y

        X = np.zeros((0, 2))
        X = vigra.taggedView(X, axistags='tc')
        y = np.zeros((0,), dtype=np.int)
        y = vigra.taggedView(y, axistags='t')

        self.Xvalid = X
        self.yvalid = y

    def testTrain(self):
        op = OpStateTrain(graph=Graph())

        op.Train.resize(2)
        op.Train[0].setValue(self.X)
        op.Train[1].setValue(self.y)

        op.Valid.resize(2)
        op.Valid[0].setValue(self.Xvalid)
        op.Valid[1].setValue(self.yvalid)

        idx = op.Classifier[0].wait()[0]
        assert isinstance(idx, int), "was {}".format(type(idx))

    def testPredict(self):
        g = Graph()
        op = OpStateTrain(graph=g)

        op.Train.resize(2)
        op.Train[0].setValue(self.X)
        op.Train[1].setValue(self.y)

        op.Valid.resize(2)
        op.Valid[0].setValue(self.Xvalid)
        op.Valid[1].setValue(self.yvalid)

        idx = op.Classifier[0].wait()[0]
        assert isinstance(idx, int), "was {}".format(type(idx))

        pred = OpStatePredict(graph=g)
        pred.Classifier.connect(op.Classifier)
        pred.Input.setValue(self.X)

        res = pred.Output[...].wait()
        np.testing.assert_array_almost_equal(res,
                                             self.y.view(np.ndarray))