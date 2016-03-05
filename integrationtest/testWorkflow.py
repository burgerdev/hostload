
import warnings
import unittest
import shutil
import tempfile

from deeplearning.workflow import Workflow
from deeplearning.workflow import RegressionWorkflow
from deeplearning.classifiers import OpStateTrain
from deeplearning.classifiers import OpStatePredict
from deeplearning.classifiers import OpSVMTrain
from deeplearning.classifiers import OpSVMPredict
from deeplearning.classifiers import OpRFTrain
from deeplearning.classifiers import OpRFPredict
from deeplearning.classifiers import OpDeepTrain
from deeplearning.classifiers import OpMLPTrain
from deeplearning.classifiers import OpMLPPredict
from deeplearning.report import OpClassificationReport
from deeplearning.report import OpRegressionReport
from deeplearning.tools import IncompatibleTargets

from pylearn2.models import mlp

from deeplearning.data.integrationdatasets import OpShuffledLinspace
from deeplearning.data.integrationdatasets import OpTarget
from deeplearning.data.integrationdatasets import OpRegTarget
from deeplearning.data.integrationdatasets import OpFeatures


config = {"source": {"class": OpShuffledLinspace,
                     "shape": (1000,)},
          "features": {"class": OpFeatures},
          "target": {"class": OpTarget},
          "train": {"class": OpStateTrain},
          "predict": {"class": OpStatePredict},
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
                w = Workflow.build(c, workingdir=self.wd)
                w.run()

        c["target"] = {"class": OpFeatures}
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
            c["class"] = RegressionWorkflow
            c["train"] = {"class": OpMLPTrain,
                          "layer_classes": (mlp.Sigmoid,),
                          "layer_sizes": (5,)}
            c["predict"] = {"class": OpMLPPredict}
            c["target"] = {"class": OpRegTarget}
            del c["report"]
            w = RegressionWorkflow.build(c, workingdir=d)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w.run()
        except:
            raise
        finally:
            shutil.rmtree(d)
