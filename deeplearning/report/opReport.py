
import os

import numpy as np

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.stype import Opaque

from deeplearning.tools.serialization import dumps
from deeplearning.split import SplitTypes


class OpReport(Operator):
    All = InputSlot(level=1)
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

    def execute(self, slot, subindex, roi, result):
        assert len(self.All) == 2, "need prediction and ground truth"
        report = dict()
        mse_all, mse_test = self._getMSE()
        report["MSE_all_data"] = mse_all
        report["MSE_test_data"] = mse_test

        fn = os.path.join(self.WorkingDir.value, "report.json")

        with open(fn, 'w') as f:
            f.write(dumps(report))
            f.write("\n")

        result[:] = True

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(slice(None))

    def _getMSE(self, samples=None):
        prediction = self.All[0][...].wait()
        expected = self.All[1][...].wait()
        m = len(prediction)

        samples = self.Description.value == SplitTypes.TEST
        prediction_test = prediction[samples]
        expected_test = expected[samples]
        n = len(prediction_test)

        mse = _mse(prediction, expected)
        mse_test = _mse(prediction_test, expected_test)
        return mse, mse_test


def _mse(a, b):
    n = len(a)
    return np.square(a-b).sum()/float(n)
