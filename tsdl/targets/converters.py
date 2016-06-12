"""
Operators that convert targets in some way.
"""

import numpy as np
import vigra

from tsdl.tools import Operator, InputSlot, OutputSlot


class OpDiscretize(Operator):
    """
    discretize an input in the range [0, 1] to values in {0, ..., N-1}
    """
    Input = InputSlot()
    NumLevels = InputSlot(value=10)

    Output = OutputSlot()

    def setupOutputs(self):
        size = self.Input.meta.shape[0]
        levels = self.NumLevels.value
        self.Output.meta.shape = (size, levels)
        self.Output.meta.dtype = np.float

    def propagateDirty(self, slot, subindex, roi):
        # TODO
        self.Output.setDirty(slice(None))

    def execute(self, slot, subindex, roi, result):
        t_start = roi.start[0]
        t_stop = roi.stop[0]
        levels = self.NumLevels.value
        # overshoot a bit to include 1.0 in last level
        boundaries = np.linspace(0, 1.0000000001, levels+1)

        input_ = self.Input[t_start:t_stop].wait()
        input_ = vigra.taggedView(input_, axistags=self.Input.meta.axistags)
        input_ = input_.withAxes('t')

        for i, j in enumerate(range(roi.start[1], roi.stop[1])):
            result[:, i] = ((input_ >= boundaries[j]) &
                            (input_ < boundaries[j+1]))


class OpClassFromOneHot(Operator):
    """
    convert (t, c) boolean matrix to (t,) index vector
    """
    Input = InputSlot()

    Output = OutputSlot()

    def setupOutputs(self):
        t_size = self.Input.meta.shape[0]
        self.Output.meta.shape = (t_size,)
        self.Output.meta.dtype = np.int

    def propagateDirty(self, slot, subindex, roi):
        # TODO
        self.Output.setDirty(slice(None))

    def execute(self, slot, subindex, roi, result):
        t_start = roi.start[0]
        t_stop = roi.stop[0]

        input_ = self.Input[t_start:t_stop, :].wait()
        result[:] = np.argmax(input_, axis=1)
