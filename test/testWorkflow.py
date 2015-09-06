
import unittest

import numpy as np
import vigra

from lazyflow.utility.testing import OpArrayPiperWithAccessCount
from lazyflow.operators import OpReorderAxes

from deeplearning.workflow import Workflow
from deeplearning.split import OpTrainTestSplit
from deeplearning.classifiers import OpStateTrain
from deeplearning.classifiers import OpStatePredict
from deeplearning.data import OpPickleCache
from deeplearning.data import OpHDF5Cache


class OpSource(OpArrayPiperWithAccessCount):
    @staticmethod
    def build(d, graph=None, parent=None, workingdir=None):
        assert "shape" in d
        data = np.random.random(size=d["shape"])
        tags = "".join([t for s, t in zip(data.shape, 'txyzc')])
        data = vigra.taggedView(data, axistags=tags)
        op = OpSource(parent=parent, graph=graph)
        op.Input.setValue(data)
        return op


class OpFeatures(OpReorderAxes):
    @staticmethod
    def build(d, graph=None, parent=None, workingdir=None):
        op = OpFeatures(parent=parent, graph=graph)
        op.AxisOrder.setValue('tc')
        return op


class OpTarget(OpArrayPiperWithAccessCount):
    @staticmethod
    def build(d, graph=None, parent=None, workingdir=None):
        op = OpTarget(parent=parent, graph=graph)
        return op

    def setupOutputs(self):
        self.Output.meta.shape = self.Input.meta.shape[0:1]
        self.Output.meta.dtype = np.int
        self.Output.meta.axistags = vigra.defaultAxistags('t')

    def execute(self, slot, subindex, roi, result):
        result[:] = 0

config = {"class": Workflow,
          "source": {"class": OpSource,
                     "shape": (1000,)},
          "features": {"class": OpFeatures},
          "target": {"class": OpTarget},
          "split": {"class": OpTrainTestSplit},
          "train": {"class": OpStateTrain},
          "classifierCache": {"class": OpPickleCache},
          "predictionCache": {"class": OpHDF5Cache},
          "predict": {"class": OpStatePredict},}


class TestWorkflow(unittest.TestCase):
    def setUp(self):
        pass

    def testBasic(self):
        try:
            w = Workflow.build(config)
            w.run()
        except:
            raise
        finally:
            pass
            # TODO remove temp dir
