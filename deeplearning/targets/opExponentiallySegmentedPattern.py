
import numpy as np

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion


class OpExponentiallySegmentedPattern(Operator):
    Input = InputSlot()
    BaselineSize = InputSlot(value=60)
    NumSegments = InputSlot(value=4)

    Output = OutputSlot()

    def setupOutputs(self):
        n = self.NumSegments.value
        self.Output.meta.assignFrom(self.Input.meta)
        self.Output.meta.shape = (self.Input.meta.shape[0], n)
        self.Output.meta.dtype = np.float32

    def execute(self, slot, subindex, roi, result):
        b = self.BaselineSize.value

        max_segment = b*2**(roi.stop[1] - 1)
        max_t = self.Output.meta.shape[0]
        new_start = (roi.start[0],)
        if roi.stop[0]+max_segment-1 <= max_t:
            new_stop = (roi.stop[0]+max_segment-1,)
            to_fill = 0
        else:
            new_stop = (max_t,)
            to_fill = roi.stop[0]+max_segment-1 - max_t
        filler = np.nan*np.zeros((to_fill,))
        new_roi = SubRegion(self.Input, start=new_start, stop=new_stop)
        x = self.Input.get(new_roi).wait()
        x = np.concatenate((x, filler))
        n_interior = roi.stop[0] - roi.start[0]

        for i, c in enumerate(range(roi.start[1], roi.stop[1])):
            s = b*2**c
            f = np.ones((s,), dtype=np.float32)/float(s)
            result[:, i] = np.convolve(x, f, mode='full')[s-1:n_interior+s-1]

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(slice(None))