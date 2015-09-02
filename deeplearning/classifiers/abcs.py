
import numpy as np

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion


class OpTrain(Operator):
    Train = InputSlot(level=1)
    Valid = InputSlot(level=1)

    Classifier = OutputSlot()

    def setupOutputs(self):
        self.Classifier.meta.shape = (1,)
        self.Classifier.meta.dtype = np.object

    def propagateDirty(self, slot, subindex, roi):
        self.Classifier.setDirty(slice(None))


class OpPredict(Operator):
    Input = InputSlot()
    Classifier = InputSlot()

    Output = OutputSlot()

    def setupOutputs(self):
        n = self.Input.meta.shape[0]
        self.Output.meta.shape = (n,)
        self.Output.meta.dtype = np.int

    def propagateDirty(self, slot, subindex, roi):
        if slot == self.Classifier:
            self.Output.setDirty(slice(None))
        else:
            a = roi.start[0]
            b = roi.stop[0]
            new_roi = SubRegion(self.Output, start=(a,), stop=(b,))
            self.Output.setDirty(new_roi)
