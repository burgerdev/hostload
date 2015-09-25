
import os
import unittest
import tempfile
import shutil

import numpy as np

from lazyflow.graph import Graph
from lazyflow.utility.testing import OpArrayPiperWithAccessCount

from deeplearning.data.caches import OpPickleCache
from deeplearning.data.caches import OpHDF5Cache


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
