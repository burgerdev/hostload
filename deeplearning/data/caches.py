"""
Caching operators. We don't want to use the caches from lazyflow right now -
partially because they are to cumbersome for our use case, partially because
we use a serialization model different from the one used for ilastik.
"""

import logging
import os
import cPickle as pkl

import numpy as np
import h5py

from deeplearning.tools import Operator, InputSlot, OutputSlot

from deeplearning.tools import Buildable

LOGGER = logging.getLogger(__name__)


class _Cache(Operator, Buildable):
    """
    a buildable cache class
    """
    Input = InputSlot()
    WorkingDir = InputSlot()
    Output = OutputSlot()

    _cached = False

    @classmethod
    def build(cls, d, parent=None, graph=None, workingdir=None):
        op = super(_Cache, cls).build(d, parent=parent, graph=graph,
                                      workingdir=workingdir)
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
    """
    caches in memory, and writes to a pickle file

    The cache is *not* filled from the file! This class is most useful for
    storing single objects for future inspection.
    """

    _filename = None
    _payload = None

    def setupOutputs(self):
        super(OpPickleCache, self).setupOutputs()
        basename = self.name + ".pkl"
        self._filename = os.path.join(self.WorkingDir.value, basename)

    def execute(self, slot, subindex, roi, result):
        if not self._cached:
            req = self.Input.get(roi)
            req.writeInto(result)
            req.block()
            self.cache(result)
            self._cached = True
            self._payload = np.zeros_like(result)
            self._payload[:] = result[:]
        else:
            result[:] = self._payload[:]

    # we don't want to quit the program if caching failed, otherwise data from
    # days of training could be lost
    # pylint: disable-msg=W0703
    def cache(self, result):
        """
        dump object to pickle file
        """
        with open(self._filename, "w") as file_:
            try:
                pkl.dump(result, file_)
            except Exception as err:
                LOGGER.error("Could not dump object!\n\t%s", err.message)


class OpHDF5Cache(_Cache):
    """
    caches in memory and in a HDF5 file

    The cache is *not* filled from file. This should be used when the caches
    output is accessed like Output[...].wait(), i.e. all at once.
    """
    _payload = None
    _filename = None
    _internal = None

    def setupOutputs(self):
        super(OpHDF5Cache, self).setupOutputs()
        basename = self.name + ".h5"
        self._filename = os.path.join(self.WorkingDir.value, basename)
        self._internal = "data"

    def _get_dataset(self, hdf5_file):
        """
        get a new hdf5 dataset (delete the old one, if existing)
        """
        if self._internal in hdf5_file:
            del hdf5_file[self._internal]

        dataset = hdf5_file.create_dataset(
            self._internal, shape=self.Input.meta.shape,
            dtype=self.Input.meta.dtype)
        return dataset

    def execute(self, slot, subindex, roi, result):
        if self._cached:
            # we have cached data
            slice_ = roi.toSlice()
            result[...] = self._payload[slice_]
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
        """
        write slice to file
        """
        with h5py.File(self._filename, "a") as file_:
            dataset = self._get_dataset(file_)
            slice_ = roi.toSlice()
            dataset[slice_] = result
