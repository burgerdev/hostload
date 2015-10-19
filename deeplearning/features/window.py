
import numpy as np
import vigra

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion

from deeplearning.tools import Buildable


class OpWindow(Operator, Buildable):
    Input = InputSlot()
    WindowSize = InputSlot()

    Output = OutputSlot()

    @classmethod
    def build(cls, config, parent=None, graph=None, workingdir=None):
        op = cls(parent=parent, graph=graph)
        if "window_size" in config:
            op.WindowSize.setValue(config["window_size"])
        return op

    @classmethod
    def get_default_config(cls):
        config = super(OpWindow, cls).get_default_config()
        config["window_size"] = 16
        return config

    def setupOutputs(self):
        assert (len(self.Input.meta.shape) <= 2 or
                np.prod(self.Input.meta.shape[1:]) == 1)
        self.Output.meta.shape = (self.Input.meta.shape[0], 1)
        self.Output.meta.axistags = vigra.defaultAxistags('tc')
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

        res_view = vigra.taggedView(result, axistags=self.Output.meta.axistags)
        res_view = res_view.withAxes('t')

        self.applyWindowFunction(x, window, res_view)

    def propagateDirty(self, slot, subindex, roi):
        roi = roi.copy()
        num_obs = self.Output.meta.shape[0]
        roi.stop[0] = min(roi.stop[0] + num_obs - 1, num_obs)
        self.Output.setDirty(roi)

    @classmethod
    def applyWindowFunction(cls, input_array, window_size, output_array):
        raise NotImplementedError()


class OpRawWindowed(OpWindow):

    @classmethod
    def applyWindowFunction(cls, input_array, window_size, output_array):
        output_array[:] = input_array[window_size-1:]


class OpDiff(OpWindow):
    @classmethod
    def build(cls, config, parent=None, graph=None, workingdir=None):
        operator = cls(parent=parent, graph=graph)
        return operator

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


class OpGaussianSmoothing(OpWindow):
    @classmethod
    def applyWindowFunction(cls, input_array, window_size, output_array):
        assert window_size % 2 == 1,\
            "window size for gaussian kernel must be odd"
        radius = (window_size - 1) / 2.0
        sigma = radius / 3
        x = np.linspace(-radius, radius, window_size)
        filter_ = np.exp(-x**2/(2*sigma**2))
        filter_ /= filter_.sum()
        smoothed = np.convolve(input_array, filter_, mode='valid')
        output_array[:] = smoothed.astype(output_array.dtype)
