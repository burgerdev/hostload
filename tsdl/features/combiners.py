"""
Operators to combine a number of feature operators.

Feature operators can be combined *horizontally*, such that all features are
available at the output, or *vertically*, such that each feature is fed into
the next operator.
"""

import numpy as np
import vigra

from lazyflow.operators.generic import OpMultiArrayStacker

from tsdl.tools import Operator, InputSlot, OutputSlot

from tsdl.tools import build_operator


class OpSimpleCombiner(Operator):
    """
    combines a list of feature operators into one (horizontally)

    operators must have slots Input and Output
    """
    Input = InputSlot()
    Output = OutputSlot()
    Valid = OutputSlot()

    @classmethod
    def build(cls, config, parent=None, graph=None, workingdir=None):
        """
        config["operators"] = <tuple of operator classes or config dicts>
        """
        to_combine = config["operators"]
        operator = cls(to_combine, parent=parent, graph=graph)
        return operator

    def __init__(self, to_combine, *args, **kwargs):
        super(OpSimpleCombiner, self).__init__(*args, **kwargs)

        operators = [build_operator(item, parent=self) for item in to_combine]

        combiner = OpMultiArrayStacker(parent=self)
        combiner.AxisFlag.setValue('c')
        combiner.Images.resize(len(operators))

        for index, operator in enumerate(operators):
            combiner.Images[index].connect(operator.Output)
            operator.Input.connect(self.Input)

        valid_combiner = OpMultiArrayStacker(parent=self)
        valid_combiner.AxisFlag.setValue('c')
        valid_operators = [op for op in operators if hasattr(op, "Valid")]
        valid_combiner.Images.resize(len(valid_operators))

        for index, operator in enumerate(valid_operators):
            valid_combiner.Images[index].connect(operator.Valid)

        self._combiner = combiner
        self._valid_combiner = valid_combiner
        self._operators = operators
        self.Output.connect(combiner.Output)

    def setupOutputs(self):
        size = self._operators[0].Input.meta.shape[0]
        self.Valid.meta.shape = (size,)
        self.Valid.meta.axistags = vigra.defaultAxistags('t')
        self.Valid.meta.dtype = np.uint8

    def execute(self, slot, subindex, roi, result):
        assert slot is self.Valid
        start = roi.start[0]
        stop = roi.stop[0]
        valid = self._valid_combiner.Output[start:stop, :].wait()
        result[:] = np.all(valid, axis=1)

    def propagateDirty(self, slot, subindex, roi):
        # Output is propagated internally, Valid should be static
        pass


class OpChain(Operator):
    """
    chains a list of feature operators (vertically)

    operators must have slots Input and Output
    """
    Input = InputSlot()
    Output = OutputSlot()
    Valid = OutputSlot()

    @classmethod
    def build(cls, config, parent=None, graph=None, workingdir=None):
        """
        config["operators"] = <tuple of operator classes or config dicts>
        """
        to_combine = config["operators"]
        operator = cls(to_combine, parent=parent, graph=graph)
        return operator

    def __init__(self, to_combine, *args, **kwargs):
        super(OpChain, self).__init__(*args, **kwargs)
        next_slot = self.Input

        operators = [build_operator(item, parent=self) for item in to_combine]

        for operator in operators:
            operator.Input.connect(next_slot)
            next_slot = operator.Output

        valid_combiner = OpMultiArrayStacker(parent=self)
        valid_combiner.AxisFlag.setValue('c')
        valid_operators = [op for op in operators if hasattr(op, "Valid")]
        valid_combiner.Images.resize(len(valid_operators))

        for index, operator in enumerate(valid_operators):
            valid_combiner.Images[index].connect(operator.Valid)

        self.Output.connect(next_slot)
        self._operators = operators
        self._valid_combiner = valid_combiner

    def setupOutputs(self):
        size = self._operators[0].Input.meta.shape[0]
        self.Valid.meta.shape = (size,)
        self.Valid.meta.axistags = vigra.defaultAxistags('t')
        self.Valid.meta.dtype = np.uint8

    def execute(self, slot, subindex, roi, result):
        assert slot is self.Valid
        start = roi.start[0]
        stop = roi.stop[0]
        valid = self._valid_combiner.Output[start:stop, :].wait()
        result[:] = np.all(valid, axis=1)

    def propagateDirty(self, slot, subindex, roi):
        # Output is propagated internally, Valid should be static
        pass
