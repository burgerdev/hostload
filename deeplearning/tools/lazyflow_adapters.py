"""
convenience wrappers for lazyflow classes
"""

import logging
import threading

import h5py

from .abcs import Buildable

from lazyflow.operator import Operator as _Operator
from lazyflow.operator import InputSlot, OutputSlot
from lazyflow.graph import Graph
from lazyflow.rtype import SubRegion
from lazyflow.operators import OpReorderAxes as _OpReorderAxes
from lazyflow.operators.ioOperators import OpStreamingHdf5Reader as _OpHdf5
from lazyflow.operators import OpArrayPiper as _OpPiper


LOGGER = logging.getLogger(__name__)


class _SetInSlotMixin(object):
    """
    we don't use setInSlot in this package
    """
    def setInSlot(self, slot, subindex, key, value):
        LOGGER.error("'setInSlot' not supported in this module")


class Operator(_SetInSlotMixin, _Operator):
    pass


class OpStreamingHdf5Reader(_SetInSlotMixin, Buildable, _OpHdf5):
    Output = OutputSlot()
    close_on_del = True

    @classmethod
    def build(cls, config, graph=None, parent=None, workingdir=None):
        op = cls(graph=graph, parent=parent)
        f = h5py.File(config["filename"], "r")
        op.Hdf5File.setValue(f)
        op.InternalPath.setValue(config["internal_path"])
        assert op.Output.ready()
        return op

    def __init__(self, *args, **kwargs):
        super(OpStreamingHdf5Reader, self).__init__(*args, **kwargs)
        self.Output.connect(self.OutputImage)

    def cleanUp(self):
        if self.close_on_del:
            self.Hdf5File.value.close()
        super(OpStreamingHdf5Reader, self).cleanUp()


class OpArrayPiper(_SetInSlotMixin, Buildable, _OpPiper):
    pass


class OpReorderAxes(_SetInSlotMixin, Buildable, _OpReorderAxes):
    pass


class OpArrayPiperWithAccessCount(OpArrayPiper):
    requests = []
    accessCount = 0
    _lock = None

    def __init__(self, *args, **kwargs):
        self._lock = threading.Lock()
        self.clear()
        super(OpArrayPiperWithAccessCount, self).__init__(*args, **kwargs)

    def execute(self, slot, subindex, roi, result):
        with self._lock:
            self.accessCount += 1
            self.requests.append(roi.copy())
        return super(OpArrayPiperWithAccessCount, self).execute(
            slot, subindex, roi, result)

    def clear(self):
        self.requests = []
        self.accessCount = 0
