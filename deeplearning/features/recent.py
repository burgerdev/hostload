"""
This module contains just one feature operator, OpRecent.
"""


import numpy as np
import vigra

from deeplearning.tools import Operator, InputSlot, OutputSlot
from deeplearning.tools import SubRegion


class OpRecent(Operator):
    """
    Provides the `window_size` most recent input values as feature channels.

    This functionality is also known as `lag operator` or `backshift operator`.
    """
    Input = InputSlot()
    WindowSize = InputSlot()

    Output = OutputSlot()
    Valid = OutputSlot()

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

        self.Valid.meta.shape = (self.Input.meta.shape[0],)
        self.Valid.meta.axistags = vigra.defaultAxistags('t')
        self.Valid.dtype = np.uint8

    def execute(self, slot, subindex, roi, result):
        if slot is self.Valid:
            return self._execute_valid(roi, result)
        elif slot is self.Output:
            return self._execute_output(roi, result)
        else:
            raise ValueError("unknown slot {}".format(slot))

    def propagateDirty(self, slot, subindex, roi):
        window = self.WindowSize.value
        max_size = self.Output.meta.shape[0]
        max_size = min(roi.stop[0] + max_size - 1, max_size)
        new_start = (roi.start[0], 0)
        new_stop = (max_size, window)
        new_roi = SubRegion(self.Output, start=new_start, stop=new_stop)
        self.Output.setDirty(new_roi)

    def _execute_valid(self, roi, result):
        window = self.WindowSize.value
        result[:] = 1
        first_valid_index = window - 1
        num_invalid = first_valid_index - roi.start[0]
        if num_invalid > 0:
            result[:num_invalid] = 0
        return result

    def _execute_output(self, roi, result):
        window = self.WindowSize.value
        padding_size = max(window - 1 - roi.start[0], 0)

        rem = tuple(self.Input.meta.shape[1:])
        new_start = (roi.start[0] - window + 1 + padding_size,) + (0,)*len(rem)
        new_stop = (roi.stop[0],) + rem
        new_roi = SubRegion(self.Input, start=new_start, stop=new_stop)

        input_ = self.Input.get(new_roi).wait()
        input_ = vigra.taggedView(input_, axistags=self.Input.meta.axistags)
        input_ = input_.withAxes('t').view(np.ndarray)

        padding = np.ones((padding_size,), dtype=np.float32) * input_[0]

        input_ = np.concatenate((padding, input_))
        n_examples = roi.stop[0] - roi.start[0]

        for i, j in enumerate(range(roi.start[1], roi.stop[1])):
            result[:, i] = input_[window-j-1:window-j-1+n_examples]
        return result
