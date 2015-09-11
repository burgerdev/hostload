
import numpy as np

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion


class OpTrain(Operator):
    Train = InputSlot(level=1)
    Valid = InputSlot(level=1)

    Classifier = OutputSlot()

    @classmethod
    def build(cls, d, parent=None, graph=None, workingdir=None):
        op = cls(parent=parent, graph=graph)
        return op

    def setupOutputs(self):
        self.Classifier.meta.shape = (1,)
        self.Classifier.meta.dtype = np.object

    def propagateDirty(self, slot, subindex, roi):
        self.Classifier.setDirty(slice(None))


class OpPredict(Operator):
    Input = InputSlot()
    Classifier = InputSlot()
    Target = InputSlot()

    Output = OutputSlot()

    @classmethod
    def build(cls, d, parent=None, graph=None, workingdir=None):
        op = cls(parent=parent, graph=graph)
        return op

    def setupOutputs(self):
        n = self.Input.meta.shape[0]
        c = self.Target.meta.shape[1]
        self.Output.meta.shape = (n, c)
        self.Output.meta.dtype = np.float

    def propagateDirty(self, slot, subindex, roi):
        if slot == self.Classifier:
            self.Output.setDirty(slice(None))
        else:
            c = self.Target.meta.shape[1]
            a = roi.start[0]
            b = roi.stop[0]
            new_roi = SubRegion(self.Output, start=(a, c), stop=(b, c))
            self.Output.setDirty(new_roi)
