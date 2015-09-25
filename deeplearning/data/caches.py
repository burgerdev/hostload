
import logging
import os
import atexit
import cPickle as pkl

import h5py
import numpy as np

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
        self._cached = False

    def execute(self, slot, subindex, roi, result):
        raise NotImplementedError()

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(roi)
        self._cached = False


class OpPickleCache(_Cache):
    def setupOutputs(self):
        super(OpPickleCache, self).setupOutputs()
        basename = self.name + ".pkl"
        self._filename = os.path.join(self.WorkingDir.value, basename)

    def execute(self, slot, subindex, roi, result):
        if not self._cached:
            req = self.Input.get(roi)
            req.writeInto(result)
            req.block()
            self.cache(roi, result)
            self._cached = True
            self._payload = np.zeros_like(result)
            self._payload[:] = result[:]
        else:
            result[:] = self._payload[:]

    def cache(self, roi, result):
        with open(self._filename, "w") as f:
            try:
                pkl.dump(result, f)
            except Exception as err:
                logger.error("Could not dump object:\n\t{}".format(str(err)))


class OpHDF5Cache(_Cache):
    def setupOutputs(self):
        super(OpHDF5Cache, self).setupOutputs()
        basename = self.name + ".h5"
        self._filename = os.path.join(self.WorkingDir.value, basename)
        self._internal = "data"

    def _getDataset(self, f):
        if self._internal in f:
            del f[self._internal]

        ds = f.create_dataset(
            self._internal, shape=self.Input.meta.shape,
            dtype=self.Input.meta.dtype)
        return ds

    def execute(self, slot, subindex, roi, result):
        if self._cached:
            # we have cached data
            s = roi.toSlice()
            result[...] = self._payload[s]
        else:
            start = np.asarray(roi.start)
            stop = np.asarray(roi.stop)
            shape = np.asarray(self.Input.meta.shape)

            req = self.Input.get(roi)
            req.writeInto(result)
            req.block()
            self.cache(roi, result)

            if np.all(start == 0) and np.all(stop == shape):
                # request is for full array, we can cache
                self._cached = True
                self._payload = np.zeros_like(result)
                self._payload[...] = result[...]

    def cache(self, roi, result):
        with h5py.File(self._filename, "a") as f:
            ds = self._getDataset(f)
            s = roi.toSlice()
            ds[s] = result
