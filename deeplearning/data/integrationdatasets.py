
import numpy as np
import vigra

from lazyflow.utility.testing import OpArrayPiperWithAccessCount
from lazyflow.operators import OpReorderAxes

from deeplearning.tools import Classification
from deeplearning.tools import Regression


TAU = 2*np.pi
MAX_SEED = 4294967295


class _BaseDataset(OpArrayPiperWithAccessCount):
    @classmethod
    def build(cls, d, graph=None, parent=None, workingdir=None):
        np.random.seed(hash(cls.__name__) % MAX_SEED)
        data = cls.createDataset(d)
        op = cls(parent=parent, graph=graph)
        op.Input.setValue(data)
        return op

    @classmethod
    def createDataset(cls, config):
        raise NotImplementedError()


class OpNoisySine(_BaseDataset):
    @classmethod
    def createDataset(cls, config):
        assert "shape" in config
        num_examples = config["shape"][0]
        num_periods = 99.9
        data = np.linspace(0, num_periods*TAU, num_examples)
        data = (np.sin(data) + 1) / 2
        noise = np.random.normal(loc=0, scale=.02, size=(num_examples,))
        data += noise
        data = vigra.taggedView(data, axistags="t")
        return data


class OpPipedTarget(OpReorderAxes, Regression):
    @classmethod
    def build(cls, d, graph=None, parent=None, workingdir=None):
        op = cls(parent=parent, graph=graph)
        op.AxisOrder.setValue('tc')
        return op


class OpShuffledLinspace(_BaseDataset):
    @classmethod
    def createDataset(cls, config):
        assert "shape" in config
        data = np.linspace(0, 1, config["shape"][0])
        data = data[np.random.permutation(len(data))]
        tags = "".join([t for s, t in zip(data.shape, 'txyzc')])
        data = vigra.taggedView(data, axistags=tags)
        return data


class OpFeatures(OpReorderAxes):
    @staticmethod
    def build(d, graph=None, parent=None, workingdir=None):
        op = OpFeatures(parent=parent, graph=graph)
        op.AxisOrder.setValue('tc')
        return op


class _OpTarget(OpArrayPiperWithAccessCount):
    @classmethod
    def build(cls, d, graph=None, parent=None, workingdir=None):
        op = cls(parent=parent, graph=graph)
        return op

    def setupOutputs(self):
        assert len(self.Input.meta.shape) == 1
        self.Output.meta.shape = (self.Input.meta.shape[0], 2)
        self.Output.meta.dtype = np.float
        self.Output.meta.axistags = vigra.defaultAxistags('tc')

    def execute(self, slot, subindex, roi, result):
        data = self.Input[roi.start[0]:roi.stop[0]].wait()
        for i, c in enumerate(range(roi.start[1], roi.stop[1])):
            result[:, i] = np.where(data > .499, c, 1-c)


class OpTarget(_OpTarget, Classification):
    pass


class OpRegTarget(OpArrayPiperWithAccessCount, Regression):
    @staticmethod
    def build(d, graph=None, parent=None, workingdir=None):
        op = OpRegTarget(parent=parent, graph=graph)
        return op

    def setupOutputs(self):
        assert len(self.Input.meta.shape) == 1
        self.Output.meta.shape = (self.Input.meta.shape[0], 1)
        self.Output.meta.dtype = np.float
        self.Output.meta.axistags = vigra.defaultAxistags('tc')

    def execute(self, slot, subindex, roi, result):
        data = self.Input[roi.start[0]:roi.stop[0]].wait()
        result[:, 0] = 1 - data
