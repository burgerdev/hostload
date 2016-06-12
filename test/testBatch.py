
import os
import unittest
import tempfile
import shutil

from tsdl.batch import run_batch

from tsdl.classifiers import OpSVMTrain
from tsdl.classifiers import OpSVMPredict

from tsdl.data.integrationdatasets import OpShuffledLinspace
from tsdl.data.integrationdatasets import OpTarget
from tsdl.data.integrationdatasets import OpFeatures

from tsdl.tools import OpArrayPiper

from tsdl.report import OpClassificationReport
from tsdl.report import OpRegressionReport


class TestBatch(unittest.TestCase):
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
        run_batch(config, self.wd, continue_on_failure=False)

    def tearDown(self):
        shutil.rmtree(self.wd)
