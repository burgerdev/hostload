"""
multi-purpose operators
"""

import numpy as np

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.operators import OpArrayPiper

from .abcs import Buildable


class IncompatibleDataset(Exception):
    """
    raise when a downstream operator is not compatible with upstream data
    """
    pass


class OpNormalize(Operator, Buildable):
    """
    perform stochastic normalization with given mean and standard deviation
    """
    Input = InputSlot()
    Mean = InputSlot(value=0.0)
    StdDev = InputSlot(value=1.0)

    Output = OutputSlot()

    @classmethod
    def build(cls, config, parent=None, graph=None, workingdir=None):
        operator = cls(parent=parent, graph=graph)
        if "mean" in config:
            operator.Mean.setValue(config["mean"])
        if "stddev" in config:
            operator.StdDev.setValue(config["stddev"])
        return operator

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)

    def execute(self, slot, subindex, roi, result):
        req = self.Input.get(roi)
        req.writeInto(result)
        req.block()
        mean = float(self.Mean.value)
        stddev = float(self.StdDev.value)
        result[:] = (result - mean) / stddev

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(slice(None))


class OpChangeDtype(OpArrayPiper, Buildable):
    """
    cast input array to different dtype
    """
    Dtype = InputSlot(value=np.float32)

    @classmethod
    def build(cls, config, parent=None, graph=None, workingdir=None):
        operator = cls(parent=parent, graph=graph)
        if "dtype" in config:
            operator.Dtype.setValue(config["dtype"])
        return operator

    def setupOutputs(self):
        super(OpChangeDtype, self).setupOutputs()
        self.Output.meta.dtype = self.Dtype.value

    def execute(self, slot, subindex, roi, result):
        out_type = self.Output.meta.dtype
        if self.Input.meta.dtype == out_type:
            super(OpChangeDtype, self).execute(slot, subindex, roi, result)
        else:
            input_ = self.Input.get(roi).wait()
            result[:] = input_.astype(out_type)
