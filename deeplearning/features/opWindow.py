
import numpy as np
import vigra

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion


class OpWindow(Operator):
    Input = InputSlot()
    WindowSize = InputSlot()

    Output = OutputSlot()

    def setupOutputs(self):
        ws = self.WindowSize.value
        self.Output.meta.shape = (self.Input.meta.shape[0] - ws + 1,)
        self.Output.meta.axistags = vigra.defaultAxistags('t')
        self.Output.meta.dtype = np.float32

    def execute(self, slot, subindex, roi, result):
        window = self.WindowSize.value
        n = roi.stop[0] - roi.start[0]
        new_stop = (roi.stop[0] + window - 1,)
        new_roi = SubRegion(self.Input, start=roi.start, stop=new_stop)
        x = self.Input.get(new_roi).wait()

        self.applyWindowFunction(x, window, result)

    def propagateDirty(self, slot, subindex, roi):
        n = self.Output.meta.shape[0]
        m = min(roi.stop[0] + n - 1, n)
        new_stop = (m,)
        new_roi = SubRegion(self.Output, start=roi.start, stop=new_stop)
        self.Output.setDirty(new_roi)

    @classmethod
    def applyWindowFunction(cls, input_array, window_size, output_array):
        raise NotImplementedError()


class OpRawWindowed(OpWindow):

    @classmethod
    def applyWindowFunction(cls, input_array, window_size, output_array):
        output_array[:] = input_array[window_size-1:]
