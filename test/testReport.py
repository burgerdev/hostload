
import os
import unittest
import tempfile
import shutil

import numpy as np

from lazyflow.graph import Graph

from deeplearning.report import OpClassificationReport
from deeplearning.report import OpRegressionReport
from deeplearning.split import SplitTypes

from deeplearning.tools.serialization import loads


class TestReport(unittest.TestCase):
    def setUp(self):
        self.wd = tempfile.mkdtemp()
        self.g = Graph()

    def tearDown(self):
        shutil.rmtree(self.wd)

    def testClassification(self):
        pred = np.asarray([[0, 1], [0, 1], [1, 0], [1, 0]])
        gt = np.asarray([[0, 1], [0, 1], [0, 1], [1, 0]])
        valid = np.ones(pred.shape[:1], dtype=np.uint8)
        desc = np.zeros((len(gt),), dtype=np.int)
        desc[:2] = SplitTypes.TRAIN
        desc[2:] = SplitTypes.TEST

        op = OpClassificationReport.build({}, graph=self.g, workingdir=self.wd)
        op.All.resize(2)
        op.All[0].setValue(pred)
        op.All[1].setValue(gt)
        op.Valid.resize(2)
        op.Valid[0].setValue(valid)
        op.Valid[1].setValue(valid)
        op.Description.setValue(desc)

        assert op.Output.value

        with open(os.path.join(self.wd, "report.json"), "r") as f:
            report = loads(f.read())

        assert report["all_true"] == 3
        assert report["test_true"] == 1
        assert report["all_false"] == 1
        assert report["test_false"] == 1

        op.All[0].setValue(1-pred)

        assert op.Output.value

        with open(os.path.join(self.wd, "report.json"), "r") as f:
            report = loads(f.read())

        assert report["all_true"] == 1
        assert report["test_true"] == 1
        assert report["all_false"] == 3
        assert report["test_false"] == 1

    def testRegression(self):
        pred = np.asarray([0, 1, 2, 3])[:, np.newaxis] / 4.0
        gt = np.asarray([0, 1, 0, 2.99])[:, np.newaxis] / 4.0
        valid = np.ones(pred.shape[:1], dtype=np.uint8)
        desc = np.zeros((len(gt),), dtype=np.int)
        desc[:2] = SplitTypes.TRAIN
        desc[2:] = SplitTypes.TEST

        d = {"levels": 5}

        op = OpRegressionReport.build(d, graph=self.g, workingdir=self.wd)
        op.All.resize(2)
        op.All[0].setValue(pred)
        op.All[1].setValue(gt)
        op.Valid.resize(2)
        op.Valid[0].setValue(valid)
        op.Valid[1].setValue(valid)
        op.Description.setValue(desc)

        assert op.Output.value

        with open(os.path.join(self.wd, "report.json"), "r") as f:
            report = loads(f.read())

        print(report)

        mse_all = np.square(pred - gt).mean()
        mse_test = np.square(pred[2:] - gt[2:]).mean()
        np.testing.assert_almost_equal(report["all_MSE"], mse_all)
        np.testing.assert_almost_equal(report["test_MSE"], mse_test)
        np.testing.assert_almost_equal(report["all_misclass"], .25)
        np.testing.assert_almost_equal(report["test_misclass"], .5)