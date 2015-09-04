
from lazyflow.operator import Operator, InputSlot, OutputSlot


class _Cache(Operator):
    Input = InputSlot()
    Output = OutputSlot()

    @classmethod
    def build(cls, d, parent=None, graph=None, workingdir=None):
        op = cls(parent=parent, graph=graph)
        return op

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)

    def execute(self, slot, subindex, roi, result):
        req = self.Input.get(roi)
        req.writeInto(result)
        req.block()
        self.cache(roi, result)

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(roi)

    def cache(self, roi, result):
        pass

class OpPickleCache(_Cache):
    pass

class OpHDF5Cache(_Cache):
    pass
