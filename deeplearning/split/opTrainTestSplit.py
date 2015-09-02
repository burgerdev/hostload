
import numpy as np
from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion


class OpTrainTestSplit(Operator):

    # expects (t, c) inputs
    #   - t indexes time slices
    #   - c indexes feature channels
    Input = InputSlot()

    # percentage of total input for testing
    TestPercentage = InputSlot(value=.1)

    # percentage of training data used for validation
    ValidPercentage = InputSlot(value=.1)

    Train = OutputSlot()
    Valid = OutputSlot()
    Test = OutputSlot()

    # 0: train, 1: valid, 2: test
    Description = OutputSlot()
    All = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpTrainTestSplit, self).__init__(*args, **kwargs)
        self.All.connect(self.Input)

    def setupOutputs(self):
        size_t = self.Input.meta.shape[0]
        shape_rem = self.Input.meta.shape[1:]

        test = self.TestPercentage.value
        valid = self.ValidPercentage.value

        size_t_train = int(np.floor(size_t * (1-test)))
        train_shape = (size_t_train,) + shape_rem
        self._all_train_shape = train_shape
        test_shape = (size_t - size_t_train,) + shape_rem

        size_t_valid = int(np.floor(size_t_train * valid))
        valid_shape = (size_t_valid,) + shape_rem
        train_shape = (size_t_train - size_t_valid,) + shape_rem

        self.Train.meta.assignFrom(self.Input.meta)
        self.Valid.meta.assignFrom(self.Input.meta)
        self.Test.meta.assignFrom(self.Input.meta)
        self.Description.meta.assignFrom(self.Input.meta)

        self.Train.meta.shape = train_shape
        self.Valid.meta.shape = valid_shape
        self.Test.meta.shape = test_shape
        self.Description.meta.dtype = np.uint8

        self._valid_offset = self.Train.meta.shape[0]
        self._test_offset = self.Train.meta.shape[0] + self.Valid.meta.shape[0]

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

        r = self.Input.get(new_roi)
        r.writeInto(result)
        r.block()

    def propagateDirty(self, slot, subindex, roi):
        pass
