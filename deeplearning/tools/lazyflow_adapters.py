"""
convenience wrappers for lazyflow classes
"""

import logging
import h5py

from .abcs import Buildable

from lazyflow.operator import Operator as _Operator
from lazyflow.operator import InputSlot, OutputSlot
from lazyflow.rtype import SubRegion
from lazyflow.operators import OpReorderAxes
from lazyflow.operators.ioOperators import OpStreamingHdf5Reader as _OpHdf5
from lazyflow.operators import OpArrayPiper as _OpPiper
from lazyflow.utility.testing import OpArrayPiperWithAccessCount


LOGGER = logging.getLogger(__name__)


class Operator(_Operator):
    def setInSlot(self, slot, subindex, key, value):
        LOGGER.error("'setInSlot' not supported in this module")


class OpStreamingHdf5Reader(_OpHdf5, Buildable, Operator):
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


class OpArrayPiper(_OpPiper, Buildable):
    @classmethod
    def build(cls, config, graph=None, parent=None, workingdir=None):
        op = cls(graph=graph, parent=parent)
        return op
