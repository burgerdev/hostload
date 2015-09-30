
import numpy as np
import vigra

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion


class OpDiscretize(Operator):
    Input = InputSlot()
    NumLevels = InputSlot(value=10)

    Output = OutputSlot()

    def setupOutputs(self):
        n = self.Input.meta.shape[0]
        l = self.NumLevels.value
        self.Output.meta.shape = (n, l)
        self.Output.meta.dtype = np.float

    def propagateDirty(self, slot, subindex, roi):
        # TODO
        self.Output.setDirty(slice(None))

    def execute(self, slot, subindex, roi, result):
        a = roi.start[0]
        b = roi.stop[0]
        l = self.NumLevels.value
        r = np.linspace(0, 1.0000000001, l+1)

        x = self.Input[a:b].wait()
        x = vigra.taggedView(x, axistags=self.Input.meta.axistags)
        x = x.withAxes('t')

        for i, c in enumerate(range(roi.start[1], roi.stop[1])):
            result[:, i] = (x >= r[c]) & (x < r[c+1])


class OpClassFromOneHot(Operator):
    Input = InputSlot()

    Output = OutputSlot()

    def setupOutputs(self):
        n = self.Input.meta.shape[0]
        self.Output.meta.shape = (n,)
        self.Output.meta.dtype = np.int

    def propagateDirty(self, slot, subindex, roi):
        # TODO
        self.Output.setDirty(slice(None))

    def execute(self, slot, subindex, roi, result):
        a = roi.start[0]
        b = roi.stop[0]

        x = self.Input[a:b, :].wait()
        result[:] = np.argmax(x, axis=1)
