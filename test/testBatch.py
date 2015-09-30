
import os
import unittest
import tempfile
import shutil

from deeplearning.batch import run_batch

from deeplearning.classifiers import OpSVMTrain
from deeplearning.classifiers import OpSVMPredict

from deeplearning.data.integrationdatasets import OpShuffledLinspace
from deeplearning.data.integrationdatasets import OpTarget
from deeplearning.data.integrationdatasets import OpFeatures

from deeplearning.data.wrappers import OpArrayPiper

from deeplearning.report import OpClassificationReport
from deeplearning.report import OpRegressionReport


class TestMLPRegression(unittest.TestCase):
    def setUp(self):
        self.wd = tempfile.mkdtemp(prefix="BatchRun_")

    def testBatch(self):
        os.mkdir(os.path.join(self.wd, "1"))
        config = {"source": {"class": OpShuffledLinspace,
                             "shape": (1000,)},
                  "preprocessing": ({"class": OpArrayPiper},),
                  "features": {"class": OpFeatures},
                  "target": {"class": OpTarget},
                  "train": {"class": OpSVMTrain},
                  "predict": {"class": OpSVMPredict},
                  "report": {"class": [OpClassificationReport,
                                       OpRegressionReport]}}
        run_batch(config, self.wd)

    def tearDown(self):
        shutil.rmtree(self.wd)
