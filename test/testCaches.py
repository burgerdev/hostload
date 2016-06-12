
import os
import unittest
import tempfile
import shutil

import numpy as np

from lazyflow.graph import Graph
from tsdl.tools.lazyflow_adapters import OpArrayPiperWithAccessCount

from tsdl.data.caches import OpPickleCache
from tsdl.data.caches import OpHDF5Cache
from tsdl.tools import OpStreamingHdf5Reader


class TestReport(unittest.TestCase):
    def setUp(self):
        self.wd = tempfile.mkdtemp()
        self.g = Graph()

    def tearDown(self):
        shutil.rmtree(self.wd)

    def testOpPickleCache(self):
        self.runWithClass(OpPickleCache)

    def testOpHDF5Cache(self):
        self.runWithClass(OpHDF5Cache)

    def runWithClass(self, cls):
        pipe = OpArrayPiperWithAccessCount(graph=self.g)
        op = cls.build({}, graph=self.g, workingdir=self.wd)
        op.Input.connect(pipe.Output)

        pipe.Input.setValue(np.random.random(size=(50, 50)))

        np.testing.assert_equal(pipe.accessCount, 0)
        op.Output[...].wait()
        np.testing.assert_equal(pipe.accessCount, 1)
        op.Output[...].wait()
        np.testing.assert_equal(pipe.accessCount, 1)
        pipe.Input.setDirty(slice(None))
        op.Output[...].wait()
        np.testing.assert_equal(pipe.accessCount, 2)

        return op

    def testOpStreamingHdf5Reader(self):
        cache = self.runWithClass(OpHDF5Cache)

        data = cache.Output[...].wait()

        config = {"class": OpStreamingHdf5Reader,
                  "filename": os.path.join(self.wd, "OpHDF5Cache.h5"),
                  "internal_path": "data"}

        op = OpStreamingHdf5Reader.build(config, graph=self.g)

        data2 = op.Output[...].wait()

        np.testing.assert_array_equal(data, data2)

        op.cleanUp()
        del op
