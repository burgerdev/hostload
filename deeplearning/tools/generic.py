
from lazyflow.operator import Operator, InputSlot, OutputSlot


class OpNormalize(Operator):
    Input = InputSlot()
    Mean = InputSlot(value=0.0)
    StdDev = InputSlot(value=1.0)

    Output = OutputSlot()

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)

    def execute(self, slot, subindex, roi, result):
        req = self.Input.get(roi)
        req.writeInto(result)
        req.block()
        result[:] = (result - float(self.Mean.value)) / float(self.StdDev.value)

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(slice(None))
