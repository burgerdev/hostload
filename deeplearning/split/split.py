
import numpy as np
import vigra

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion


class SplitTypes:
    TRAIN = 0
    VALID = 1
    TEST = 2

class OpTrainTestSplit(Operator):

    # expects (t, c) inputs
    #   - t indexes time slices
    #   - c indexes feature channels
    Input = InputSlot(level=1)

    # percentage of total input for testing
    TestPercentage = InputSlot(value=.1)

    # percentage of training data used for validation
    ValidPercentage = InputSlot(value=.1)

    Train = OutputSlot(level=1)
    Valid = OutputSlot(level=1)
    Test = OutputSlot(level=1)

    # see SplitTypes
    Description = OutputSlot()
    All = OutputSlot(level=1)

    def __init__(self, *args, **kwargs):
        super(OpTrainTestSplit, self).__init__(*args, **kwargs)
        self.All.connect(self.Input)

        def _onSizeChanged(slot, old_size, new_size):
            for s in (self.Train, self.Valid, self.Test, self.All):
                s.resize(new_size)

        self.Input.notifyResized(_onSizeChanged)

    def setupOutputs(self):
        test = self.TestPercentage.value
        valid = self.ValidPercentage.value

        for subindex, subslot in enumerate(self.Input):
            size_t = subslot.meta.shape[0]
            shape_rem = subslot.meta.shape[1:]

            size_t_train = int(np.floor(size_t * (1-test)))
            train_shape = (size_t_train,) + tuple(shape_rem)
            test_shape = (size_t - size_t_train,) + tuple(shape_rem)

            size_t_valid = int(np.floor(size_t_train * valid))
            valid_shape = (size_t_valid,) + shape_rem
            train_shape = (size_t_train - size_t_valid,) + shape_rem

            for slot in (self.Train, self.Valid, self.Test, self.All):
                slot[subindex].meta.assignFrom(subslot.meta)

            self.Train[subindex].meta.shape = train_shape
            self.Valid[subindex].meta.shape = valid_shape
            self.Test[subindex].meta.shape = test_shape

        self.Description.meta.shape = (size_t,)
        self.Description.meta.dtype = np.uint8
        self.Description.meta.axistags = vigra.defaultAxistags('t')

        self._valid_offset = self.Train[0].meta.shape[0]
        self._test_offset = (self.Train[0].meta.shape[0] +
                             self.Valid[0].meta.shape[0])
        self._all_train_shape = train_shape

    def execute(self, slot, subindex, roi, result):
        if slot == self.Description:
            x = np.arange(roi.start[0], roi.stop[0])
            valid = (x >= self._valid_offset) & (x < self._test_offset)
            test = x >= self._test_offset
            result[:] = 0
            result[valid, ...] = 1
            result[test, ...] = 2
            return

        if slot == self.Train:
            offset = 0
            new_start = tuple(roi.start)
            new_stop = tuple(roi.stop)
        elif slot == self.Valid:
            offset = self._valid_offset
        elif slot == self.Test:
            offset = self._test_offset

        new_start = (roi.start[0] + offset,) + tuple(roi.start[1:])
        new_stop = (roi.stop[0] + offset,) + tuple(roi.stop[1:])
        new_roi = SubRegion(self.Input, start=new_start, stop=new_stop)

        r = self.Input[subindex[0]].get(new_roi)
        r.writeInto(result)
        r.block()

    def propagateDirty(self, slot, subindex, roi):
        # FIXME
        pass

    @staticmethod
    def build(d, parent=None, graph=None, workingdir=None):
        op = OpTrainTestSplit(parent=parent, graph=graph)
        if "test" in d:
            op.TestPercentage.setValue(d["test"])
        if "valid" in d:
            op.ValidPercentage.setValue(d["valid"])
        return op
