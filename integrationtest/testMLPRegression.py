
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
from deeplearning.classifiers import OpMLPTrain
from deeplearning.classifiers import OpMLPPredict
from deeplearning.data import OpPickleCache
from deeplearning.data import OpHDF5Cache
from deeplearning.features import OpRecent
from deeplearning.report import OpRegressionReport
from deeplearning.targets import OpExponentiallySegmentedPattern

from pylearn2.models import mlp

TAU = 2*np.pi


class OpSource(OpArrayPiperWithAccessCount):
    @staticmethod
    def build(d, graph=None, parent=None, workingdir=None):
        assert "shape" in d
        num_examples = d["shape"][0]
        num_periods = 99.9
        data = np.linspace(0, num_periods*TAU, num_examples)
        data = (np.sin(data) + 1) / 2
        np.random.seed(1)
        noise = np.random.normal(loc=0, scale=.02, size=(num_examples,))
        data += noise
        data = vigra.taggedView(data, axistags="t")
        op = OpSource(parent=parent, graph=graph)
        op.Input.setValue(data)
        return op


config = {"class": Workflow,
          "source": {"class": OpSource,
                     "shape": (10000,)},
          "features": {"class": OpRecent, "window_size": 30},
          "target": {"class": OpExponentiallySegmentedPattern,
                     "baseline_size": 10,
                     "num_segments": 1},
          "split": {"class": OpTrainTestSplit},
          "train": {"class": OpMLPTrain},
          "classifierCache": {"class": OpPickleCache},
          "predict": {"class": OpMLPPredict},
          "predictionCache": {"class": OpHDF5Cache},
          "report": {"class": OpRegressionReport,
                     "levels": 50}}


class TestMLPRegression(object):
    def setUp(self):
        pass

    def testRun(self):
        d = tempfile.mkdtemp(prefix="MLPReg_")
        try:
            c = config.copy()
            c["train"] = {"class": OpMLPTrain,
                          "layer_classes": (mlp.Sigmoid, mlp.Sigmoid),
                          "layer_sizes": (20, 10)}
            c["predict"] = {"class": OpMLPPredict}
            w = Workflow.build(c, workingdir=d)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w.run()
            self.__verify(w)
        except:
            w = None
            raise
        finally:
            import sys
            sys.stderr.write("testMLP: {}\n".format(d))
            # TODO remove dir
            # shutil.rmtree(d)
            return w

    def __verify(self, w):
        pass


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    test = TestMLPRegression()
    test.setUp()
    w = test.testRun()

    pred = w._predictionCache.Output[...].wait()
    target = w._target.Output[...].wait()

    plt.plot(target, 'b')
    plt.plot(pred, 'r+')
    plt.legend(("ground truth", "prediction"))
    plt.show()
