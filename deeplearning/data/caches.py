
import logging
import os
import atexit
import cPickle as pkl
import h5py

from lazyflow.operator import Operator, InputSlot, OutputSlot

logger = logging.getLogger(__name__)


class _Cache(Operator):
    Input = InputSlot()
    WorkingDir = InputSlot()
    Output = OutputSlot()

    @classmethod
    def build(cls, d, parent=None, graph=None, workingdir=None):
        op = cls(parent=parent, graph=graph)
        op.WorkingDir.setValue(workingdir)
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
    def setupOutputs(self):
        super(OpPickleCache, self).setupOutputs()
        basename = self.name + ".pkl"
        fn = os.path.join(self.WorkingDir.value, basename)
        self._file = open(fn, "w")
        atexit.register(self._file.close)

    def cache(self, roi, result):
        try:
            pkl.dump(result, self._file)
        except Exception as err:
            logger.error("Could not dump object:\n\t{}".format(str(err)))

class OpHDF5Cache(_Cache):
    def setupOutputs(self):
        super(OpHDF5Cache, self).setupOutputs()
        basename = self.name + ".h5"
        internal = "data"
        fn = os.path.join(self.WorkingDir.value, basename)
        self._file = h5py.File(fn, "w")
        atexit.register(self._file.close)

        if internal in self._file:
            del self._file[internal]

        self._ds = self._file.create_dataset(
            internal, shape=self.Input.meta.shape,
            dtype=self.Input.meta.dtype)

    def cache(self, roi, result):
        s = roi.toSlice()
        self._ds[s] = result
