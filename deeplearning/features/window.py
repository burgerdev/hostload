"""
Operators acting on a window of the input.

The `OpWindow` class is the base class for all feature operators that provide
windowed functionality (e.g. filter operators). These operators are tailored
for time series: the window is taken from the past to the current index so that
no information from the future leaks into the current feature.
"""

import numpy as np
import vigra

from deeplearning.tools import Operator, InputSlot, OutputSlot
from deeplearning.tools import SubRegion


class OpWindow(Operator):
    """
    base class for windowed features

    The feature is computed by applying the child class'
    `apply_window_function` on a window of the recent `WindowSize`
    observations.
    """
    Input = InputSlot()
    WindowSize = InputSlot()

    Output = OutputSlot()
    Valid = OutputSlot()

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

        self.Valid.meta.shape = (self.Input.meta.shape[0],)
        self.Valid.meta.axistags = vigra.defaultAxistags('t')
        self.Valid.meta.dtype = np.uint8

    def execute(self, slot, subindex, roi, result):
        window = self.WindowSize.value

        if slot is self.Output:
            padding_size = max(window - 1 - roi.start[0], 0)
            padding = np.zeros((padding_size,), dtype=np.float32)
            new_start = (roi.start[0] - window + 1 + padding_size,)
            new_stop = (roi.stop[0],)
            new_roi = SubRegion(self.Input, start=new_start, stop=new_stop)
            input_data = self.Input.get(new_roi).wait()
            input_data = vigra.taggedView(input_data,
                                          axistags=self.Input.meta.axistags)
            input_data = input_data.withAxes('t').view(np.ndarray)
            input_data = np.concatenate((padding, input_data))

            res_view = vigra.taggedView(result,
                                        axistags=self.Output.meta.axistags)
            res_view = res_view.withAxes('t')

            self.apply_window_function(input_data, window, res_view)
        elif slot is self.Valid:
            result[:] = 1
            first_valid_index = window - 1
            num_invalid = first_valid_index - roi.start[0]
            if num_invalid > 0:
                result[:num_invalid] = 0

    def propagateDirty(self, slot, subindex, roi):
        roi = roi.copy()
        num_obs = self.Output.meta.shape[0]
        roi.stop[0] = min(roi.stop[0] + num_obs - 1, num_obs)
        self.Output.setDirty(roi)

    @classmethod
    def apply_window_function(cls, input_array, window_size, output_array):
        """
        compute the feature on given data

        Child classes must override this.

        @input input_array requested input + (window_size-1) older observations
        @input window_size the window size in use
        @input output_array preallocated output
        """
        raise NotImplementedError()


class OpRawWindowed(OpWindow):
    """
    OpArrayPiper emulation with windows
    """
    @classmethod
    def apply_window_function(cls, input_array, window_size, output_array):
        output_array[:] = input_array[window_size-1:]


class OpDiff(OpWindow):
    """
    derivative operator: of diff[k] = input[k] - input[k-1]
    uses numpy.diff
    """
    @classmethod
    def build(cls, config, parent=None, graph=None, workingdir=None):
        operator = cls(parent=parent, graph=graph)
        return operator

    def __init__(self, *args, **kwargs):
        super(OpDiff, self).__init__(*args, **kwargs)
        self.WindowSize.setValue(2)

    @classmethod
    def apply_window_function(cls, input_array, window_size, output_array):
        output_array[:] = np.diff(input_array)


class OpFairness(OpWindow):
    """
    fairness operator (see [1])
    [1]: Di, Sheng, Derrick Kondo, and Walfredo Cirne. "Host load prediction in
    a Google compute cloud with a Bayesian model." Proceedings of the
    International Conference on High Performance Computing, Networking, Storage
    and Analysis. IEEE Computer Society Press, 2012.

    http://mescal.imag.fr/membres/sheng.di/download/SC2012-loadprediction.pdf
    """
    @classmethod
    def apply_window_function(cls, input_array, window_size, output_array):
        sum_filter = np.ones((window_size,), dtype=np.float32)/window_size
        squares = np.square(input_array)
        sum_of_squares = np.convolve(squares, sum_filter, mode='valid')
        sums = np.convolve(input_array, sum_filter, mode='valid')
        squares_of_sums = np.square(sums)
        output_array[:] = squares_of_sums/sum_of_squares
