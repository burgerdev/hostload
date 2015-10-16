
import unittest

import numpy as np
import vigra

from deeplearning.classifiers.mlp import PCAWeightInitializer
from deeplearning.classifiers.mlp import LeastSquaresWeightInitializer

from lazyflow.graph import Graph

from pylearn2.models.mlp import MLP
from pylearn2.models.mlp import Sigmoid


class TestInitializers(unittest.TestCase):
    def setUp(self):
        n = 1000
        d = 5
        X = np.random.random(size=(n, d))
        y = np.sum(X, axis=1, keepdims=True)

        # self.X = vigra.taggedView(X, axistags='tc')
        # self.y = vigra.taggedView(y, axistags='tc')
        self.X = X
        self.y = y
        self.n = n
        self.d = d

    def testPCA(self):
        init = PCAWeightInitializer.build({}, graph=Graph())
        init.Input.resize(2)
        init.Input[0].setValue(self.X)
        init.Input[1].setValue(self.y)

        def norm(x):
            return np.sqrt(np.square(x).sum())

        for k in (2, 3, 5, 7):
            layer = Sigmoid(layer_name="a", irange=0, dim=k)
            # layer needs to be initialized by MLP first
            MLP(layers=[layer], nvis=self.d)
            np.random.seed(123)
            init.init_layer(layer, nvis=self.d, nhid=k)

            weights = layer.get_weights()
            np.testing.assert_array_equal(weights.shape, (self.d, k))
            assert np.abs(weights).sum() > 0

            if k <= self.d:
                for i in range(k):
                    np.testing.assert_almost_equal(norm(weights[:, i]), 1)
                    for j in range(i+1, k):
                        dot = np.dot(weights[:, i], weights[:, j])
                        np.testing.assert_almost_equal(dot, 0)

    def testLSF(self):
        init = LeastSquaresWeightInitializer.build({}, graph=Graph())
        init.Input.resize(2)
        init.Input[0].setValue(self.X)
        init.Input[1].setValue(self.y)

        for k in (2, 3, 5, 7):
            layer = Sigmoid(layer_name="a", irange=0, dim=k)
            # layer needs to be initialized by MLP first
            MLP(layers=[layer], nvis=self.d)
            np.random.seed(123)
            init.init_layer(layer, nvis=self.d, nhid=k)

            weights = layer.get_weights()
            np.testing.assert_array_equal(weights.shape, (self.d, k))
            assert np.abs(weights).sum() > 0

