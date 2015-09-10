
import unittest

import numpy as np
import vigra

from lazyflow.graph import Graph
from lazyflow.utility.testing import OpArrayPiperWithAccessCount

from deeplearning.data import OpDataset


class TestOpDataset(unittest.TestCase):
    def setUp(self):
        X = np.random.random(size=(1000, 5))
        X = vigra.taggedView(X, axistags='tc')
        y = np.random.randint(0, 2, size=(1000,))
        y = vigra.taggedView(y, axistags='t')
        self.X = X
        self.y = y

        self.g = Graph()

        self.pipeData = OpArrayPiperWithAccessCount(graph=self.g)
        self.pipeTarget = OpArrayPiperWithAccessCount(graph=self.g)

        self.op = OpDataset(graph=self.g)
        self.op.Input.resize(2)
        self.op.Input[0].connect(self.pipeData.Output)
        self.op.Input[1].connect(self.pipeTarget.Output)

        self.pipeData.Input.setValue(self.X)
        self.pipeTarget.Input.setValue(self.y)

    def test_adjust_for_viewer(self):
        self.op.adjust_for_viewer(self.X)

    def test_get_num_examples(self):
        n = self.op.get_num_examples()
        np.testing.assert_equal(n, self.X.shape[0])

    def test_has_targets(self):
        assert self.op.has_targets()

    def test_iterator(self):
        k = 5
        i = self.op.iterator(mode='sequential', batch_size=k)

        for batch in i:
            np.testing.assert_equal(len(batch), k)

        i = self.op.iterator(mode='sequential', num_batches=k)
        batches = [b for b in i]
        np.testing.assert_equal(len(batches), k)

        i = self.op.iterator(mode='sequential', num_batches=k)
        i = iter(i)
        n = i.next()
        assert isinstance(n, np.ndarray)

        i = self.op.iterator(mode='sequential', num_batches=k,
                             return_tuple=True)
        i = iter(i)
        n = i.next()
        assert isinstance(n, tuple)
        np.testing.assert_equal(len(n), 1)
