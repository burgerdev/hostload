
import numpy as np
import vigra

from lazyflow.utility.testing import OpArrayPiperWithAccessCount

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
