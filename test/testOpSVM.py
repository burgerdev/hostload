
import unittest

import numpy as np
import vigra

from sklearn.svm import SVC

from lazyflow.graph import Graph

from deeplearning.classifiers import OpSVMTrain
from deeplearning.classifiers import OpSVMPredict


class TestOpSVM(unittest.TestCase):
    def setUp(self):
        X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        X = vigra.taggedView(X, axistags='tc')
        y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        y = vigra.taggedView(y, axistags='tc')

        self.X = X
        self.y = y

        X = np.zeros((0, 2))
        X = vigra.taggedView(X, axistags='tc')
        y = np.zeros((0, 2))
        y = vigra.taggedView(y, axistags='tc')

        self.Xvalid = X
        self.yvalid = y

    def testTrain(self):
        op = OpSVMTrain(graph=Graph())

        op.Train.resize(2)
        op.Train[0].setValue(self.X)
        op.Train[1].setValue(self.y)

        op.Valid.resize(2)
        op.Valid[0].setValue(self.Xvalid)
        op.Valid[1].setValue(self.yvalid)

        svc = op.Classifier[0].wait()[0]
        assert isinstance(svc, SVC), "was {}".format(type(svc))

    def testPredict(self):
        g = Graph()
        op = OpSVMTrain(graph=g)

        op.Train.resize(2)
        op.Train[0].setValue(self.X)
        op.Train[1].setValue(self.y)

        op.Valid.resize(2)
        op.Valid[0].setValue(self.Xvalid)
        op.Valid[1].setValue(self.yvalid)

        svc = op.Classifier[0].wait()[0]
        assert isinstance(svc, SVC), "was {}".format(type(svc))

        pred = OpSVMPredict(graph=g)
        pred.Classifier.connect(op.Classifier)
        pred.Input.setValue(self.X)
        pred.Target.connect(op.Train[1])

        res = pred.Output[...].wait()
        np.testing.assert_array_equal(res, self.y.view(np.ndarray))
