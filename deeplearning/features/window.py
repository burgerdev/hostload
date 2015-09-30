
import numpy as np
import vigra

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion


class OpWindow(Operator):
    Input = InputSlot()
    WindowSize = InputSlot()

    Output = OutputSlot()

    @classmethod
    def build(cls, config, parent=None, graph=None, workingdir=None):
        op = cls(parent=parent, graph=graph)
        return op

    def setupOutputs(self):
        self.Output.meta.shape = (self.Input.meta.shape[0],)
        self.Output.meta.axistags = vigra.defaultAxistags('t')
        self.Output.meta.dtype = np.float32

    def execute(self, slot, subindex, roi, result):
        window = self.WindowSize.value
        padding_size = max(window - 1 - roi.start[0], 0)
        padding = np.zeros((padding_size,), dtype=np.float32)
        new_start = (roi.start[0] - window + 1 + padding_size,)
        new_stop = (roi.stop[0],)
        new_roi = SubRegion(self.Input, start=new_start, stop=new_stop)
        x = self.Input.get(new_roi).wait()
        x = vigra.taggedView(x, axistags=self.Input.meta.axistags)
        x = x.withAxes('t').view(np.ndarray)
        x = np.concatenate((padding, x))

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


class OpDiff(OpWindow):
    def __init__(self, *args, **kwargs):
        super(OpDiff, self).__init__(*args, **kwargs)
        self.WindowSize.setValue(2)

    @classmethod
    def applyWindowFunction(cls, input_array, window_size, output_array):
        output_array[:] = np.diff(input_array)


class OpFairness(OpWindow):
    @classmethod
    def applyWindowFunction(cls, input_array, window_size, output_array):
        sum_filter = np.ones((window_size,), dtype=np.float32)/window_size
        squares = np.square(input_array)
        sum_of_squares = np.convolve(squares, sum_filter, mode='valid')
        sums = np.convolve(input_array, sum_filter, mode='valid')
        squares_of_sums = np.square(sums)
        output_array[:] = squares_of_sums/sum_of_squares
