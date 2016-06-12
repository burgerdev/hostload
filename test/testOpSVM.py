
import unittest

import numpy as np
import vigra

from sklearn.svm import SVC
from sklearn.svm import SVR

from lazyflow.graph import Graph

from tsdl.classifiers import OpSVMTrain
from tsdl.classifiers import OpSVMPredict


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
        op = OpSVMTrain.build(dict(), graph=Graph())

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

        pred = OpSVMPredict.build(dict(), graph=g)
        pred.Classifier.connect(op.Classifier)
        pred.Input.setValue(self.X)
        pred.Target.connect(op.Train[1])

        res = pred.Output[...].wait()
        np.testing.assert_array_equal(res, self.y.view(np.ndarray))

        pred.Classifier.disconnect()
        pred.Classifier.setValue([None])
        pred.Input.setValue(None)
        pred.Input.setValue(self.X)
        with self.assertRaises(ValueError):
            pred.Output[...].wait()


class TestOpSVR(unittest.TestCase):
    def setUp(self):
        n = 100
        np.random.seed(1)
        X = np.random.random(size=(n, 2))
        X = vigra.taggedView(X, axistags='tc')
        y = X.sum(axis=1).withAxes(*'tc')

        self.X = X
        self.y = y

        X = np.zeros((0, 2))
        X = vigra.taggedView(X, axistags='tc')
        y = np.zeros((0, 1))
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

        svr = op.Classifier[0].wait()[0]
        assert isinstance(svr, SVR), "was {}".format(type(svr))

    def testPredict(self):
        g = Graph()
        op = OpSVMTrain(graph=g)

        op.Train.resize(2)
        op.Train[0].setValue(self.X)
        op.Train[1].setValue(self.y)

        op.Valid.resize(2)
        op.Valid[0].setValue(self.Xvalid)
        op.Valid[1].setValue(self.yvalid)

        svr = op.Classifier[0].wait()[0]
        assert isinstance(svr, SVR), "was {}".format(type(svc))

        pred = OpSVMPredict(graph=g)
        pred.Classifier.connect(op.Classifier)
        pred.Input.setValue(self.X)
        pred.Target.connect(op.Train[1])

        res = pred.Output[...].wait()
        np.testing.assert_array_almost_equal(res, self.y.view(np.ndarray),
                                             decimal=1)
