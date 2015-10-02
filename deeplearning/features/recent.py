
import numpy as np
import vigra

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion

from deeplearning.tools import Buildable


class OpRecent(Operator, Buildable):
    Input = InputSlot()
    WindowSize = InputSlot()

    Output = OutputSlot()

    @classmethod
    def build(cls, d, parent=None, graph=None, workingdir=None):
        op = cls(parent=parent, graph=graph)
        op.WindowSize.setValue(d["window_size"])
        return op

    def setupOutputs(self):
        window = self.WindowSize.value
        self.Output.meta.shape = (self.Input.meta.shape[0], window)
        self.Output.meta.axistags = vigra.defaultAxistags('tc')
        self.Output.meta.dtype = self.Input.meta.dtype

    def execute(self, slot, subindex, roi, result):
        window = self.WindowSize.value
        padding_size = max(window - 1 - roi.start[0], 0)

        rem = tuple(self.Input.meta.shape[1:])
        new_start = (roi.start[0] - window + 1 + padding_size,) + (0,)*len(rem)
        new_stop = (roi.stop[0],) + rem
        new_roi = SubRegion(self.Input, start=new_start, stop=new_stop)

        x = self.Input.get(new_roi).wait()
        x = vigra.taggedView(x, axistags=self.Input.meta.axistags)
        x = x.withAxes('t').view(np.ndarray)

        padding = np.ones((padding_size,), dtype=np.float32) * x[0]

        x = np.concatenate((padding, x))
        n_examples = roi.stop[0] - roi.start[0]

        for i, c in enumerate(range(roi.start[1], roi.stop[1])):
            result[:, i] = x[window-c-1:window-c-1+n_examples]

    def propagateDirty(self, slot, subindex, roi):
        window = self.WindowSize.value
        n = self.Output.meta.shape[0]
        m = min(roi.stop[0] + n - 1, n)
        new_start = (roi.start[0], 0)
        new_stop = (m, window)
        new_roi = SubRegion(self.Output, start=new_start, stop=new_stop)
        self.Output.setDirty(new_roi)
