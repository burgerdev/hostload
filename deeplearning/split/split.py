"""
Module contains the class OpTrainTestSplit, which splits data by time.
"""

import numpy as np
import vigra

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion

from deeplearning.tools import Buildable


class SplitTypes(object):  # pylint: disable=R0903,W0232
    """
    enum of split types

    (pylint disable is for message "too few public methods")
    """
    TRAIN = 0
    VALID = 1
    TEST = 2


class OpTrainTestSplit(Buildable, Operator):
    """
    splits by time into train, test and validation set

    all level 1 slots expect their subslots to contain (features, targets)

    t=0                                             t=t_max
    [            Train          | Validation |    Test    ]

    """

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

    # private
    _valid_offset = None
    _test_offset = None
    _all_train_shape = None

    def __init__(self, *args, **kwargs):
        super(OpTrainTestSplit, self).__init__(*args, **kwargs)
        self.All.connect(self.Input)

        # ignore unused arguments -> pylint: disable=W0613
        def _on_size_changed(changed_slot, old_size, new_size):
            """
            resize output to match input's size
            """
            for output_slot in (self.Train, self.Valid, self.Test, self.All):
                output_slot.resize(new_size)

        self.Input.notifyResized(_on_size_changed)

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
            req_indices = np.arange(roi.start[0], roi.stop[0])
            valid = ((req_indices >= self._valid_offset) &
                     (req_indices < self._test_offset))
            test = req_indices >= self._test_offset
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

        request = self.Input[subindex[0]].get(new_roi)
        request.writeInto(result)
        request.block()

    def propagateDirty(self, slot, subindex, roi):
        # FIXME
        pass

    @classmethod
    def build(cls, config, parent=None, graph=None, workingdir=None):
        op = super(OpTrainTestSplit, cls).build(config,
                                                parent=parent, graph=graph)
        # disable false positives
        op.TestPercentage.setValue(op._test)  # pylint: disable=E1101,W0212
        op.ValidPercentage.setValue(op._valid)  # pylint: disable=E1101,W0212
        return op

    @classmethod
    def get_default_config(cls):
        config = super(OpTrainTestSplit, cls).get_default_config()
        config["test"] = 0.1
        config["valid"] = 0.1
        return config
