"""
provides the "exponentially segmented pattern" target proposed by Kondo et.al.
"""
import numpy as np
import vigra

from deeplearning.tools import Operator, InputSlot, OutputSlot
from deeplearning.tools import SubRegion
from deeplearning.tools import OpReorderAxes

from deeplearning.tools import Regression
from deeplearning.tools import Buildable


class OpExponentiallySegmentedPattern(Operator, Regression, Buildable):
    """
    compute the mean value over exponentially growing windows
    """
    Input = InputSlot()
    BaselineSize = InputSlot(value=60)
    NumSegments = InputSlot(value=4)

    Output = OutputSlot()
    Valid = OutputSlot()

    _Input = OutputSlot()

    @classmethod
    def build(cls, d, parent=None, graph=None, workingdir=None):
        op = cls(parent=parent, graph=graph)
        op.BaselineSize.setValue(d["baseline_size"])
        op.NumSegments.setValue(d["num_segments"])
        return op

    def __init__(self, *args, **kwargs):
        super(OpExponentiallySegmentedPattern, self).__init__(*args, **kwargs)
        reorder = OpReorderAxes(parent=self)
        reorder.AxisOrder.setValue('t')
        reorder.Input.connect(self.Input)
        self._Input.connect(reorder.Output)

    def setupOutputs(self):
        num_examples = self._Input.meta.shape[0]
        num_segments = self.NumSegments.value
        self.Output.meta.shape = (num_examples, num_segments)
        self.Output.meta.axistags = vigra.defaultAxistags('tc')
        self.Output.meta.dtype = np.float32

        self.Valid.meta.shape = (num_examples,)
        self.Valid.meta.axistags = vigra.defaultAxistags('t')
        self.Valid.meta.dtype = np.uint8

    def execute(self, slot, subindex, roi, result):
        if slot is self.Valid:
            return self._execute_valid(roi, result)
        elif slot is self.Output:
            return self._execute_output(roi, result)
        else:
            raise ValueError("unknown slot {}".format(slot))

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(slice(None))

    def _execute_valid(self, roi, result):
        """
        for which time steps do we have enough future values?
        """
        baseline = self.BaselineSize.value
        max_t = self.Output.meta.shape[0]

        max_segment = baseline*2**(self.Output.meta.shape[1] - 1)
        result[:] = 1
        num_invalid = (max_segment - 1) - (max_t - roi.stop[0])
        if num_invalid > 0:
            result[max(roi.stop[0] - roi.start[0] - num_invalid, 0):] = 0
        return result

    def _execute_output(self, roi, result):
        """
        compute feature
        """
        baseline = self.BaselineSize.value
        max_t = self.Output.meta.shape[0]

        max_segment = baseline*2**(roi.stop[1] - 1)

        if roi.stop[0]+max_segment-1 <= max_t:
            new_stop = (roi.stop[0]+max_segment-1,)
            to_fill = 0
        else:
            new_stop = (max_t,)
            to_fill = roi.stop[0]+max_segment-1 - max_t
        new_roi = SubRegion(self.Input, start=(roi.start[0],), stop=new_stop)
        input_ = self._Input.get(new_roi).wait()
        input_ = np.concatenate((input_, np.zeros((to_fill,))))

        for i, j in enumerate(range(roi.start[1], roi.stop[1])):
            size = baseline*2**j
            filter_ = np.ones((size,), dtype=np.float32)/float(size)
            intermediate = np.convolve(input_, filter_, mode='full')
            result[:, i] = intermediate[size-1:roi.stop[0]-roi.start[0]+size-1]
        return result
