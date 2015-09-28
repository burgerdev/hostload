
import h5py

from lazyflow.operators.ioOperators import OpStreamingHdf5Reader as _OpHdf5
from lazyflow.operator import OutputSlot


class OpStreamingHdf5Reader(_OpHdf5):
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

    def __del__(self):
        if self.close_on_del:
            self.Hdf5File.value.close()
