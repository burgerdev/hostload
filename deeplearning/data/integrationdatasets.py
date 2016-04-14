"""
datasets used for integration testing

This is just a collection of useful datasets which seemed useful for testing
purposes. The classes in here should not be relied upon in production mode.
"""

import numpy as np
import vigra

from deeplearning.tools import OpArrayPiperWithAccessCount
from deeplearning.tools import OpReorderAxes
from deeplearning.tools import OutputSlot

from deeplearning.tools import Classification
from deeplearning.tools import Regression

from .rk4 import default_mackey_glass_series


TAU = 2*np.pi
MAX_SEED = 4294967295


# pylint seems to be somewhat broken regarding mixins
# pylint: disable=C0103
# pylint: disable=C0111

class _BaseDataset(OpArrayPiperWithAccessCount):
    """
    base class for all integration datasets
    """
    @classmethod
    def build(cls, d, graph=None, parent=None, workingdir=None):
        rng = np.random.RandomState(hash(cls.__name__) % MAX_SEED)
        op = super(_BaseDataset, cls).build(d, graph=graph, parent=parent,
                                            workingdir=workingdir)
        data = op.create_dataset(d, rng)
        op.Input.setValue(data)
        return op

    @classmethod
    def get_default_config(cls):
        config = super(_BaseDataset, cls).get_default_config()
        config["shape"] = (10000,)
        return config

    def create_dataset(self, config, rng):
        """
        create a dataset with given config and RNG

        overridden in subclasses
        """
        raise NotImplementedError()


class OpNoisySine(_BaseDataset):
    """
    a sine curve with added noise
    """

    def create_dataset(self, config, rng):
        num_examples = self._shape[0]
        num_periods = 99.9
        data = np.linspace(0, num_periods*TAU, num_examples)
        data = (np.sin(data) + 1) / 2
        noise = rng.normal(loc=0, scale=.02, size=(num_examples,))
        data += noise
        data = vigra.taggedView(data, axistags="t")
        return data


class OpShuffledLinspace(_BaseDataset):
    """
    equally spaced data from [0, 1], shuffled
    """

    def create_dataset(self, config, rng):
        data = np.linspace(0, 1, self._shape[0])
        data = data[rng.permutation(len(data))]
        data = vigra.taggedView(data, axistags='t')
        data = data.astype(np.float32)
        return data


class OpFeatures(OpReorderAxes):
    """
    pass on features with added Valid slot
    """
    Valid = OutputSlot()

    @classmethod
    def build(cls, *args, **kwargs):
        op = super(OpFeatures, cls).build(*args, **kwargs)
        op.AxisOrder.setValue('tc')
        return op

    def setupOutputs(self):
        super(OpFeatures, self).setupOutputs()
        # self.Output.meta.dtype = self.Input.meta.dtype
        self.Valid.meta.shape = self.Output.meta.shape[:1]
        self.Valid.meta.axistags = vigra.defaultAxistags('t')
        self.Valid.meta.dtype = np.uint8

    def execute(self, slot, subindex, roi, result):
        if slot is not self.Valid:
            super(OpFeatures, self).execute(slot, subindex, roi, result)
        else:
            result[:] = 1


class OpTarget(Classification, OpArrayPiperWithAccessCount):
    """
    basic classification target
    """
    Valid = OutputSlot()

    def setupOutputs(self):
        assert len(self.Input.meta.shape) == 1
        self.Output.meta.shape = (self.Input.meta.shape[0], 2)
        self.Output.meta.dtype = np.float32
        self.Output.meta.axistags = vigra.defaultAxistags('tc')
        self.Valid.meta.shape = self.Output.meta.shape[:1]
        self.Valid.meta.axistags = vigra.defaultAxistags('t')
        self.Valid.meta.dtype = np.uint8

    def execute(self, slot, subindex, roi, result):
        if slot is not self.Valid:
            data = self.Input[roi.start[0]:roi.stop[0]].wait()
            for i, channel in enumerate(range(roi.start[1], roi.stop[1])):
                result[:, i] = np.where(data > .499, channel, 1-channel)
        else:
            result[:] = 1


class OpRegTarget(Regression, OpArrayPiperWithAccessCount):
    """
    basic regression target
    """
    Valid = OutputSlot()

    def setupOutputs(self):
        # assert len(self.Input.meta.shape) == 1
        self.Output.meta.shape = (self.Input.meta.shape[0], 1)
        self.Output.meta.dtype = np.float32
        self.Output.meta.axistags = vigra.defaultAxistags('tc')
        self.Valid.meta.shape = self.Output.meta.shape[:1]
        self.Valid.meta.axistags = vigra.defaultAxistags('t')
        self.Valid.meta.dtype = np.uint8

    def execute(self, slot, subindex, roi, result):
        if slot is self.Valid:
            result[:] = 1
            return
        data = self.Input[roi.start[0]:roi.stop[0]].wait()
        result[:, 0] = 1 - data


class OpRandomUnitSquare(_BaseDataset):
    """
    random data from the (2D) unit square
    """
    @classmethod
    def get_default_config(cls):
        config = _BaseDataset.get_default_config()
        config["shape"] = (10000, 2)
        return config

    def create_dataset(self, config, rng):
        data = rng.rand(*self._shape)
        data = vigra.taggedView(data, axistags="tc")
        return data


class OpXORTarget(OpRegTarget):
    """
    The result of (kinda) XORing channel 0 and 1

      xor_cont(a, b) := 1 - (1 - a - b)^2
    """
    def execute(self, slot, subindex, roi, result):
        if slot is self.Valid:
            result[:] = 1
            return
        data = self.Input[roi.start[0]:roi.stop[0], :].wait()
        result[:, 0] = 1 - np.square(1 - data.sum(axis=1))


class OpNormTarget(OpRegTarget):
    """
    euclidean norm of features
    """
    def execute(self, slot, subindex, roi, result):
        if slot is self.Valid:
            result[:] = 1
            return
        data = self.Input[roi.start[0]:roi.stop[0]].wait()
        result[:, 0] = np.sqrt(np.square(data).sum(axis=1)/2.0)


class OpMackeyGlass(_BaseDataset):
    """
    dataset from Mackey-Glass function

    performance of "last known state estimator":
        baseline -> MSE
        8        -> 0.000199561830787
        16       -> 0.000720119728291
        32       -> 0.00263267743529
        64       -> 0.00935371769061
    """

    def create_dataset(self, config, rng):
        data = default_mackey_glass_series()
        upper = data.max()
        lower = data.min()
        data = (data - lower)/(upper - lower)
        data = vigra.taggedView(data, axistags="tc").withAxes('t')
        return data
