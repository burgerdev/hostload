"""
Reporting operators.

The operators in this module are intended to be the used as the end of an
operator chain in a workflow. They
  * trigger the start of computation
  * evaluate the computed results
Evalutation depends on the type of problem (i.e. regression or classification),
each problem type has its own set of quality metrics.
"""

import os
from collections import OrderedDict

import numpy as np

from deeplearning.tools import Operator, InputSlot, OutputSlot

from deeplearning.tools.serialization import dumps
from deeplearning.tools import Classification
from deeplearning.tools import Regression
from deeplearning.tools import Buildable
from deeplearning.split import SplitTypes


class _DictReport(object):
    """
    convenience class to aggregate reports
    """
    def __init__(self):
        self._report = dict()

    def add(self, item, value, category=None):
        """
        add a key value pair to the report
        """
        if category is not None:
            key = "{}_{}".format(category, item)
        else:
            key = item
        self._report[key] = value

    def get_report(self):
        """
        get an ordered representation of the report
        """
        ordered = OrderedDict()
        for key in sorted(self._report.keys()):
            ordered[key] = self._report[key]
        return ordered


class _OpReport(Operator, Buildable):
    """
    base class for reports

    A report needs
      * all data (input and target)
        -> "All"
      * a description of the data (training, validation or test)
        -> "Description"
      * an indicator of sample validity (samples could have missing data, or
        features could be invalid close to the experiments boundaries)
        -> "Valid"
      * an output slot which starts computation and returns a report
    """
    All = InputSlot(level=1)
    Valid = InputSlot(level=1)
    Description = InputSlot()
    WorkingDir = InputSlot()
    Output = OutputSlot()

    @classmethod
    def build(cls, d, parent=None, graph=None, workingdir=None):
        op = super(_OpReport, cls).build(d, parent=parent, graph=graph,
                                         workingdir=workingdir)
        op.WorkingDir.setValue(workingdir)
        return op

    def setupOutputs(self):
        self.Output.meta.shape = (1,)
        self.Output.meta.dtype = np.bool

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(slice(None))

    def execute(self, slot, subindex, roi, result):
        assert len(self.All) == 2, "need prediction and ground truth"
        report = _DictReport()

        self._get_report(report)

        filename = os.path.join(self.WorkingDir.value, "report.json")
        with open(filename, 'w') as file_:
            file_.write(dumps(report.get_report()))
            file_.write("\n")

        result[:] = True

    def _get_report(self, report):
        """
        fill report with key value pairs
        """
        raise NotImplementedError()


class OpRegressionReport(_OpReport, Regression):
    """
    report for regression problems

    The "Levels" slot is used to calculate the metric used by Kondo et.al,
    evaluating the difference between target and prediction by simply observing
    whether they fall in the same histogram bin.
    """
    Levels = InputSlot(optional=True)

    @classmethod
    def build(cls, d, parent=None, graph=None, workingdir=None):
        op = super(OpRegressionReport, cls).build(d, parent=parent,
                                                  graph=graph,
                                                  workingdir=workingdir)
        return op

    @classmethod
    def get_default_config(cls):
        conf = super(OpRegressionReport, cls).get_default_config()
        conf["levels"] = 50
        return conf

    def _get_report(self, report):
        if self.Levels.ready():
            levels = self.Levels.value
        else:
            levels = self._levels
        prediction = self.All[0][...].wait()
        expected = self.All[1][...].wait()
        valid = np.logical_and(self.Valid[0][...].wait(),
                               self.Valid[1][...].wait()).astype(np.bool)

        test_samples = self.Description.value == SplitTypes.TEST
        all_samples = np.ones_like(test_samples)

        for sample, which in zip((all_samples, test_samples), ("all", "test")):
            selection = sample & valid

            pred_for_sample = prediction[selection]
            exp_for_sample = expected[selection]
            mse = _mse(pred_for_sample, exp_for_sample)
            misclass = _misclass_from_regression(pred_for_sample,
                                                 exp_for_sample,
                                                 levels)
            report.add("MSE", mse, which)
            report.add("misclass", misclass, which)
            report.add("num_examples", sample.sum(), which)
            report.add("num_valid_examples", selection.sum(), which)

        report.add("levels", levels)


class OpClassificationReport(_OpReport, Classification):
    """
    Report for classification problems
    """
    def _get_report(self, report):
        prediction = np.argmax(self.All[0][...].wait(), axis=1)
        expected = np.argmax(self.All[1][...].wait(), axis=1)
        samples = self.Description.value == SplitTypes.TEST

        for sample, which in zip((samples, np.ones_like(samples)),
                                 ('test', 'all')):
            pred_for_sample = prediction[sample]
            exp_for_sample = expected[sample]
            false = (pred_for_sample != exp_for_sample).sum()
            true = len(exp_for_sample) - false
            for name, value in zip(("false", "true"), (false, true)):
                report.add(name, value, which)


def _mse(ground_truth, prediction):
    """
    mean squared error
    """
    num_observations = len(ground_truth)
    return np.square(ground_truth-prediction).sum()/float(num_observations)


def _misclass_from_regression(ground_truth, prediction, num_levels):
    """
    turn a regression into a classification and return the misclassified ratio
    """
    def inside(array, interval):
        """
        for each value, return if its inside the interval
          [interval[0], interval[1]]
        (both ends are included)
        """
        return (array >= interval[0]) & (array <= interval[1])

    bins = np.linspace(0, 1, num_levels+1)
    num_obs = len(ground_truth)

    correct = 0
    for i in range(num_levels):
        correct += (inside(ground_truth, bins[i:i+2]) &
                    inside(prediction, bins[i:i+2])).sum()

    return (num_obs - correct)/float(num_obs)
