
import warnings
import unittest
import shutil
import tempfile

import numpy as np
import vigra

from lazyflow.utility.testing import OpArrayPiperWithAccessCount
from lazyflow.operators import OpReorderAxes

from deeplearning.workflow import Workflow
from deeplearning.split import OpTrainTestSplit
from deeplearning.classifiers import OpStateTrain
from deeplearning.classifiers import OpStatePredict
from deeplearning.classifiers import OpSVMTrain
from deeplearning.classifiers import OpSVMPredict
from deeplearning.classifiers import OpRFTrain
from deeplearning.classifiers import OpRFPredict
from deeplearning.classifiers import OpDeepTrain
from deeplearning.classifiers import OpMLPTrain
from deeplearning.classifiers import OpMLPPredict
from deeplearning.data import OpPickleCache
from deeplearning.data import OpHDF5Cache
from deeplearning.report import OpClassificationReport
from deeplearning.report import OpRegressionReport
from deeplearning.tools import Classification
from deeplearning.tools import Regression
from deeplearning.tools import IncompatibleTargets

from pylearn2.models import mlp


class OpSource(OpArrayPiperWithAccessCount):
    @staticmethod
    def build(d, graph=None, parent=None, workingdir=None):
        assert "shape" in d
        data = np.linspace(0, 1, d["shape"][0])
        np.random.seed(420)
        data = data[np.random.permutation(len(data))]
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


config = {"class": Workflow,
          "source": {"class": OpSource,
                     "shape": (1000,)},
          "features": {"class": OpFeatures},
          "target": {"class": OpTarget},
          "split": {"class": OpTrainTestSplit},
          "train": {"class": OpStateTrain},
          "classifierCache": {"class": OpPickleCache},
          "predict": {"class": OpStatePredict},
          "predictionCache": {"class": OpHDF5Cache},
          "report": {"class": OpClassificationReport}}


class TestWorkflow(unittest.TestCase):
    def setUp(self):
        self.wd = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.wd)

    def testBasic(self):
        w = Workflow.build(config, workingdir=self.wd)
        w.run()

    def testIncompatible(self):
        c = config.copy()

        def foo():
            with self.assertRaises(IncompatibleTargets):
                Workflow.build(c, workingdir=self.wd)

        c["target"] = {"class": _OpTarget}
        foo()
        c["target"] = {"class": OpRegTarget}
        c["train"] = {"class": OpRFTrain}
        foo()
        c["train"] = {"class": OpSVMTrain}
        c["predict"] = {"class": OpRFPredict}
        foo()
        c["predict"] = {"class": OpSVMPredict}
        c["report"] = {"class": OpClassificationReport}
        foo()
        c["report"] = {"class": OpRegressionReport}
        Workflow.build(c, workingdir=self.wd)

    def testSVM(self):
        d = tempfile.mkdtemp()
        try:
            c = config.copy()
            c["train"] = {"class": OpSVMTrain}
            c["predict"] = {"class": OpSVMPredict}
            w = Workflow.build(c, workingdir=d)
            w.run()
        except:
            raise
        finally:
            shutil.rmtree(d)

    def testRF(self):
        d = tempfile.mkdtemp()
        try:
            c = config.copy()
            c["train"] = {"class": OpRFTrain}
            c["predict"] = {"class": OpRFPredict}
            w = Workflow.build(c, workingdir=d)
            w.run()
        except:
            raise
        finally:
            shutil.rmtree(d)

    def testDNN(self):
        d = tempfile.mkdtemp()
        try:
            c = config.copy()
            c["train"] = {"class": OpDeepTrain,
                          "num_hidden_layers": 2,
                          "size_hidden_layers": (2, 2)}
            c["predict"] = {"class": OpMLPPredict}
            w = Workflow.build(c, workingdir=d)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w.run()
        except:
            raise
        finally:
            shutil.rmtree(d)

    def testMLP(self):
        d = tempfile.mkdtemp()
        try:
            c = config.copy()
            c["train"] = {"class": OpMLPTrain,
                          "layer_classes": (mlp.Sigmoid,),
                          "layer_sizes": (5,)}
            c["predict"] = {"class": OpMLPPredict}
            w = Workflow.build(c, workingdir=d)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w.run()
        except:
            raise
        finally:
            shutil.rmtree(d)

    def testMLPReg(self):
        d = tempfile.mkdtemp()
        try:
            c = config.copy()
            c["train"] = {"class": OpMLPTrain,
                          "layer_classes": (mlp.Sigmoid,),
                          "layer_sizes": (5,)}
            c["predict"] = {"class": OpMLPPredict}
            c["source"] = {"class": OpRegSource,
                           "shape": (1000,)}
            c["target"] = {"class": OpRegTarget}
            c["report"] = {"class": OpRegressionReport, "levels": 10}
            w = Workflow.build(c, workingdir=d)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w.run()
        except:
            raise
        finally:
            shutil.rmtree(d)


class OpRegSource(OpArrayPiperWithAccessCount):
    @staticmethod
    def build(d, graph=None, parent=None, workingdir=None):
        assert "shape" in d
        data = np.linspace(0, 1, d["shape"][0])
        tags = "".join([t for s, t in zip(data.shape, 'txyzc')])
        data = vigra.taggedView(data, axistags=tags)
        op = OpSource(parent=parent, graph=graph)
        op.Input.setValue(data)
        return op


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
