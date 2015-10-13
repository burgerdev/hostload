
import unittest

import numpy as np
import vigra

from lazyflow.graph import Graph

from deeplearning.classifiers import OpStateTrain
from deeplearning.classifiers import OpStatePredict
from deeplearning.classifiers import OpStateTrainRegression
from deeplearning.classifiers import OpStatePredictRegression


class TestOpState(unittest.TestCase):
    def setUp(self):
        X = np.array([[1.2, 5], [2.2, 8], [3.1, 15], [3.9, 17]])
        X = vigra.taggedView(X, axistags='tc')
        y = np.eye(4)
        y = vigra.taggedView(y, axistags='tc')

        self.X = X
        self.y = y

        X = np.zeros((0, 2))
        X = vigra.taggedView(X, axistags='tc')
        y = np.zeros((0, 4))
        y = vigra.taggedView(y, axistags='tc')

        self.Xvalid = X
        self.yvalid = y

    def testTrain(self):
        op = OpStateTrain.build(dict(), graph=Graph())

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

        pred = OpStatePredict.build(dict(), graph=g)
        pred.Classifier.connect(op.Classifier)
        pred.Input.setValue(self.X)
        pred.Target.connect(op.Train[1])

        res = pred.Output[...].wait()
        np.testing.assert_array_almost_equal(res,
                                             np.eye(4, k=1))

    def testRegression(self):
        g = Graph()
        op = OpStateTrainRegression.build(dict(index=1), graph=g)

        op.Train.resize(2)
        op.Train[0].setValue(self.X)
        op.Train[1].setValue(self.X[:, 1:2])

        op.Valid.resize(2)
        op.Valid[0].setValue(self.Xvalid)
        op.Valid[1].setValue(self.Xvalid[:, 1:2])

        idx = op.Classifier[0].wait()[0]
        assert isinstance(idx, int), "was {}".format(type(idx))

        pred = OpStatePredictRegression.build({}, graph=g)
        pred.Classifier.connect(op.Classifier)
        pred.Input.setValue(self.X)
        pred.Target.connect(op.Train[1])

        res = pred.Output[...].wait()
        exp = self.X[:, 1:2].view(np.ndarray)
        np.testing.assert_array_almost_equal(res, exp)
