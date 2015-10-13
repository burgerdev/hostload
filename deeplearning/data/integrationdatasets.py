
import numpy as np
import vigra

from lazyflow.utility.testing import OpArrayPiperWithAccessCount
from lazyflow.operators import OpReorderAxes

from deeplearning.tools import Classification
from deeplearning.tools import Regression
from deeplearning.tools import Buildable

from .rk4 import default_mackey_glass_series


TAU = 2*np.pi
MAX_SEED = 4294967295


class _BaseDataset(OpArrayPiperWithAccessCount, Buildable):
    @classmethod
    def build(cls, d, graph=None, parent=None, workingdir=None):
        rng = np.random.RandomState(hash(cls.__name__) % MAX_SEED)
        op = super(_BaseDataset, cls).build(d, graph=graph, parent=parent,
                                            workingdir=workingdir)
        data = op.create_dataset(d, rng)
        op.Input.setValue(data)
        return op

    def create_dataset(self, config, rng):
        raise NotImplementedError()


class OpNoisySine(_BaseDataset):
    @classmethod
    def get_default_config(cls):
        config = _BaseDataset.get_default_config()
        config["shape"] = (10000,)
        return config

    def create_dataset(self, config, rng):
        num_examples = self._shape[0]
        num_periods = 99.9
        data = np.linspace(0, num_periods*TAU, num_examples)
        data = (np.sin(data) + 1) / 2
        noise = rng.normal(loc=0, scale=.02, size=(num_examples,))
        data += noise
        data = vigra.taggedView(data, axistags="t")
        return data


class OpPipedTarget(OpReorderAxes, Regression, Buildable):
    @classmethod
    def build(cls, d, graph=None, parent=None, workingdir=None):
        op = cls(parent=parent, graph=graph)
        op.AxisOrder.setValue('tc')
        return op


class OpShuffledLinspace(_BaseDataset):
    @classmethod
    def get_default_config(cls):
        config = _BaseDataset.get_default_config()
        config["shape"] = (10000, 2)
        return config

    def create_dataset(self, config, rng):
        data = np.linspace(0, 1, self._shape[0])
        data = data[rng.permutation(len(data))]
        data = vigra.taggedView(data, axistags='t')
        return data


class OpFeatures(OpReorderAxes, Buildable):
    @staticmethod
    def build(d, graph=None, parent=None, workingdir=None):
        op = OpFeatures(parent=parent, graph=graph)
        op.AxisOrder.setValue('tc')
        return op


class _OpTarget(OpArrayPiperWithAccessCount, Buildable):
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


class OpRegTarget(OpArrayPiperWithAccessCount, Regression, Buildable):
    @classmethod
    def build(cls, d, graph=None, parent=None, workingdir=None):
        op = cls(parent=parent, graph=graph)
        return op

    def setupOutputs(self):
        # assert len(self.Input.meta.shape) == 1
        self.Output.meta.shape = (self.Input.meta.shape[0], 1)
        self.Output.meta.dtype = np.float
        self.Output.meta.axistags = vigra.defaultAxistags('tc')

    def execute(self, slot, subindex, roi, result):
        data = self.Input[roi.start[0]:roi.stop[0]].wait()
        result[:, 0] = 1 - data


class OpRandomUnitSquare(_BaseDataset):
    @classmethod
    def get_default_config(cls):
        config = _BaseDataset.get_default_config()
        config["shape"] = (10000, 2)
        return config

    def create_dataset(self, config, rng):
        data = rng.rand(*self._shape)
        data = vigra.taggedView(data, axistags="tc")
        return data


class OpNormTarget(OpRegTarget):
    def execute(self, slot, subindex, roi, result):
        data = self.Input[roi.start[0]:roi.stop[0]].wait()
        result[:, 0] = np.sqrt(np.square(data).sum(axis=1)/2.0)


class OpMackeyGlass(_BaseDataset):
    def create_dataset(self, config, rng):
        data = default_mackey_glass_series()
        upper = data.max()
        lower = data.min()
        data = (data - lower)/(upper - lower)
        data = vigra.taggedView(data, axistags="tc").withAxes('t')
        return data
