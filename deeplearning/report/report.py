
import os
from collections import OrderedDict

import numpy as np

from lazyflow.operator import Operator, InputSlot, OutputSlot

from deeplearning.tools.serialization import dumps
from deeplearning.tools import Classification
from deeplearning.tools import Regression
from deeplearning.tools import Buildable
from deeplearning.split import SplitTypes


class _OpReport(Operator, Buildable):
    All = InputSlot(level=1)
    Valid = InputSlot(level=1)
    Description = InputSlot()
    WorkingDir = InputSlot()
    Output = OutputSlot()

    @classmethod
    def build(cls, d, parent=None, graph=None, workingdir=None):
        op = cls(parent=parent, graph=graph)
        op.WorkingDir.setValue(workingdir)
        return op

    def setupOutputs(self):
        self.Output.meta.shape = (1,)
        self.Output.meta.dtype = np.bool

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(slice(None))


class OpRegressionReport(_OpReport, Regression):
    Levels = InputSlot()

    @classmethod
    def build(cls, d, parent=None, graph=None, workingdir=None):
        my_d = {"levels": 50}
        my_d.update(d)
        op = cls(parent=parent, graph=graph)
        op.WorkingDir.setValue(workingdir)
        op.Levels.setValue(my_d["levels"])
        return op

    def execute(self, slot, subindex, roi, result):
        assert len(self.All) == 2, "need prediction and ground truth"
        report = self._getReport()

        fn = os.path.join(self.WorkingDir.value, "report.json")

        with open(fn, 'w') as f:
            f.write(dumps(report))
            f.write("\n")

        result[:] = True

    def _getReport(self):
        report = dict()
        prediction = self.All[0][...].wait()
        expected = self.All[1][...].wait()
        valid_features = self.Valid[0][...].wait()
        valid_target = self.Valid[1][...].wait()
        valid = np.logical_and(valid_features, valid_target).astype(np.bool)

        samples = self.Description.value == SplitTypes.TEST
        levels = self.Levels.value

        num_all_valid = valid.sum()
        num_test = samples.sum()
        num_all = len(prediction)
        prediction = prediction[valid]
        expected = expected[valid]
        print(samples.shape, valid.shape)
        samples = samples[valid]
        num_test_valid = samples.sum()

        prediction_test = prediction[samples]
        expected_test = expected[samples]

        report["all_MSE"] = _mse(prediction, expected)
        report["test_MSE"] = _mse(prediction_test, expected_test)

        report["all_misclass"] = _misclass_from_regression(prediction, expected,
                                                           levels)
        report["test_misclass"] = _misclass_from_regression(prediction_test,
                                                            expected_test,
                                                            levels)

        report["test_num_examples"] = num_test
        report["test_num_valid_examples"] = num_test_valid
        report["all_num_examples"] = num_all
        report["all_num_valid_examples"] = num_all_valid

        orderedReport = OrderedDict()
        orderedReport["levels"] = levels

        for key in sorted(report.keys()):
            orderedReport[key] = report[key]

        return orderedReport


class OpClassificationReport(_OpReport, Classification):
    def execute(self, slot, subindex, roi, result):
        assert len(self.All) == 2, "need prediction and ground truth"
        report = dict()

        prediction = np.argmax(self.All[0][...].wait(), axis=1)
        expected = np.argmax(self.All[1][...].wait(), axis=1)
        samples = self.Description.value == SplitTypes.TEST

        for s, which in zip((samples, np.ones_like(samples)),
                            ('test', 'all')):
            p = prediction[s]
            e = expected[s]
            values, names = self._getReport(p, e)
            for name, value in zip(names, values):
                key = "{}_{}".format(which, name)
                report[key] = value

        orderedReport = OrderedDict()
        for key in sorted(report.keys()):
            orderedReport[key] = report[key]

        fn = os.path.join(self.WorkingDir.value, "report.json")
        with open(fn, 'w') as f:
            f.write(dumps(orderedReport))
            f.write("\n")

        result[:] = True

    def _getReport(self, prediction, expected):
        f = (prediction != expected).sum()
        t = (prediction == expected).sum()
        return ((f, t), ("false", "true"))


def _mse(a, b):
    n = len(a)
    return np.square(a-b).sum()/float(n)

def _misclass_from_regression(a, b, l):
    def inside(x, interval):
        return (x >= interval[0]) & (x <= interval[1])

    p = np.linspace(0, 1, l+1)
    n = len(a)

    correct = 0
    for i in range(l):
        correct += (inside(a, p[i:i+2]) & inside(b, p[i:i+2])).sum()

    return (n - correct)/float(n)
