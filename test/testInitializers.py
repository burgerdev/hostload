
import unittest

import numpy as np
import vigra

from deeplearning.classifiers.mlp_init import PCAWeightInitializer
from deeplearning.classifiers.mlp_init import LeastSquaresWeightInitializer
from deeplearning.classifiers.mlp_init import OpForwardLayers
from deeplearning.classifiers.mlp_init import OptimalInitializer
from deeplearning.classifiers.mlp_init import GridInitializer

from lazyflow.graph import Graph

from pylearn2.models.mlp import MLP
from pylearn2.models.mlp import Sigmoid
from pylearn2.models.mlp import Linear


class TestInitializers(unittest.TestCase):
    def setUp(self):
        n = 1000
        d = 5
        X = np.random.random(size=(n, d))
        y = np.sum(X, axis=1, keepdims=True)

        self.X = X
        self.y = y
        self.n = n
        self.d = d

    def testPCA(self):
        init = PCAWeightInitializer.build({}, graph=Graph())
        init.Input.resize(2)
        init.Input[0].setValue(self.X)
        init.Input[1].setValue(self.y)

        for k in (2, 4, 6, 8):
            layer = Sigmoid(layer_name="a", irange=0, dim=k)
            # layer needs to be initialized by MLP first
            MLP(layers=[layer], nvis=self.d)
            np.random.seed(123)
            init.init_layer(layer, nvis=self.d, nhid=k)

            weights = layer.get_weights()
            np.testing.assert_array_equal(weights.shape, (self.d, k))
            assert np.abs(weights).sum() > 0

            if k <= 2*self.d:
                for i in range(k):
                    for j in range(i+2, k, 2):
                        dot = np.dot(weights[:, i], weights[:, j])
                        np.testing.assert_almost_equal(dot, 0, decimal=4)

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

    def testForward(self):
        k = 2
        layer = Linear(layer_name='y', irange=0, dim=k)
        MLP(layers=[layer], nvis=self.d)

        op = OpForwardLayers([layer], graph=Graph())
        op.Input.setValue(self.X)
        np.testing.assert_array_equal(op.Output.meta.shape, (self.n, k))
        not_zero = np.ones(op.Output.meta.shape, dtype=op.Output.meta.dtype)
        req = op.Output[...]
        req.writeInto(not_zero)
        req.block()
        np.testing.assert_array_equal(not_zero, 0)

        op = OpForwardLayers([], graph=Graph())
        op.Input.setValue(self.X)
        np.testing.assert_array_equal(op.Output.meta.shape, self.X.shape)
        out = op.Output[...].wait()
        np.testing.assert_array_almost_equal(out, self.X)

    def testOptimalInit(self):
        x = np.asarray([0, 0, 1, 1, .5])
        y = np.asarray([0, 1, 0, 1, .5])
        vol = np.zeros((5, 2), dtype=np.float32)
        vol[:, 0] = x
        vol[:, 1] = y
        vol = vigra.taggedView(vol, axistags='tc')

        z = x*(1-y) + y*(1-x)
        z = vigra.taggedView(z[:, np.newaxis], axistags='tc')

        nvis = 2
        ncent = 5
        layers = [Sigmoid(layer_name='bumps', irange=0, dim=2*nvis*ncent),
                  Sigmoid(layer_name='cents', irange=0, dim=ncent),
                  Linear(layer_name='out', irange=0, dim=1)]
        mlp = MLP(layers=layers, nvis=nvis)

        init = OptimalInitializer.build({}, graph=Graph())
        init.Data.setValue(vol)
        init.Target.setValue(z)
        init.init_model(mlp)

        op = OpForwardLayers(layers[:], graph=Graph())
        op.Input.setValue(vol)

        z_pred = op.Output[..., 0].wait().squeeze()
        assert max(z_pred[0], z_pred[3]) < z_pred[4]
        assert min(z_pred[1], z_pred[2]) > z_pred[4]

    def testGridInit1(self):
        num_dim = 2
        num_bins = 3

        x = np.asarray([0, 0, 1, 1, .5])
        y = np.asarray([0, 1, 0, 1, .5])
        vol = np.zeros((5, num_dim), dtype=np.float32)
        vol[:, 0] = x
        vol[:, 1] = y
        vol = vigra.taggedView(vol, axistags='tc')
        target = np.zeros((vol.shape[0], 1))
        target = vigra.taggedView(target, axistags='tc')

        """
        layers = [Sigmoid(layer_name='bumps', irange=0, dim=2*nvis*ncent),
                  Sigmoid(layer_name='cents', irange=0, dim=ncent),
                  Linear(layer_name='out', irange=0, dim=1)]
        """

        layers = [Linear(layer_name='1d', irange=0, dim=num_dim*num_bins),
                  Sigmoid(layer_name='cents', irange=0, dim=1),
                  Linear(layer_name='out', irange=0, dim=1)]

        mlp = MLP(layers=layers, nvis=num_dim)

        init = GridInitializer.build({}, graph=Graph())
        init.Data.setValue(vol)
        init.Target.setValue(target)
        init.init_model(mlp)

        op = OpForwardLayers(layers[:1], graph=Graph())
        op.Input.setValue(vol)

        output = op.Output[:, :num_bins].wait()
        computed = np.argmin(np.abs(output), axis=1)
        expected = np.asarray([0, 0, 2, 2, 1])

        np.testing.assert_array_equal(computed, expected)
