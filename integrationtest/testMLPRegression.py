
import warnings
import tempfile

import numpy as np
import vigra

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

from integrationdatasets import OpNoisySine


config = {"class": Workflow,
          "source": {"class": OpNoisySine,
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
        self.wd = tempfile.mkdtemp(prefix="MLPReg_")

    def tearDown(self):
        import sys
        sys.stderr.write("testMLP: {}\n".format(self.wd))
        # TODO remove dir
        # shutil.rmtree(d)
        

    def testRun(self):
        c = config.copy()
        c["train"] = {"class": OpMLPTrain,
                      "layer_classes": (mlp.Sigmoid, mlp.Sigmoid),
                      "layer_sizes": (20, 10)}
        c["predict"] = {"class": OpMLPPredict}
        w = Workflow.build(c, workingdir=self.wd)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.run()
        self.__verify(w)
        return w

    def __verify(self, w):
        pass


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    test = TestMLPRegression()
    test.setUp()
    try:
        w = test.testRun()
    finally:
        test.tearDown()

    pred = w._predictionCache.Output[...].wait()
    target = w._target.Output[...].wait()

    plt.plot(target, 'b')
    plt.plot(pred, 'r+')
    plt.legend(("ground truth", "prediction"))
    plt.show()
