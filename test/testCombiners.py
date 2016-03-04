
import unittest

import numpy as np
import vigra

from lazyflow.graph import Graph
from deeplearning.tools import OpArrayPiper

from deeplearning.features import OpSimpleCombiner
from deeplearning.features import OpChain
from deeplearning.features import OpFairness
from deeplearning.features import OpMean

from deeplearning.tools.generic import OpChangeDtype


class TestCombiners(unittest.TestCase):
    def setUp(self):
        array = np.random.random(size=(1000,)).astype(np.float32)
        array = vigra.taggedView(array, axistags='t')
        array = array.withAxes(*'tc')
        self.array = array

    def testOpSimpleCombiner(self):
        op1_config = {"class": OpMean, "window_size": 5}
        op2_config = {"class": OpFairness, "window_size": 7}

        op1 = OpMean.build(op1_config, graph=Graph())
        op2 = OpFairness.build(op2_config, graph=Graph())

        config = {"class": OpSimpleCombiner,
                  "operators": (OpArrayPiper, op1_config, op2_config)}
        comb = OpSimpleCombiner.build(config, graph=Graph())

        for op in (op1, op2, comb):
            op.Input.setValue(self.array)

        for s in comb._combiner.Images:
            assert s.ready(), str(s)

        assert comb.Output.ready(), "operator not set up correctly"

        piped = comb.Output[:, 0].wait()
        np.testing.assert_array_equal(piped, self.array.view(np.ndarray))

        mean = comb.Output[:, 1].wait()
        mean_gt = op1.Output[...].wait()
        np.testing.assert_array_equal(mean, mean_gt)

        fair = comb.Output[:, 2].wait()
        fair_gt = op2.Output[...].wait()
        np.testing.assert_array_equal(fair, fair_gt)

        valid = comb.Valid[...].wait()
        np.testing.assert_array_equal(valid[:6], 0)
        np.testing.assert_array_equal(valid[6:], 1)

    def testOpChain(self):
        n = 15
        op1_config = OpChangeDtype
        op2_config = {"class": OpMean, "window_size": 2}

        chain = OpChain.build({"class": OpChain,
                               "operators": (op1_config, op2_config)},
                              graph=Graph())

        array = np.arange(n, dtype=np.int)
        array = vigra.taggedView(array, axistags='t')

        chain.Input.setValue(array)
        out = chain.Output[...].wait()

        assert out.dtype == np.float32

        expected = np.concatenate(((0,), np.arange(n-1)+0.5))
        expected = expected.astype(np.float32)
        expected = expected[:, np.newaxis]
        np.testing.assert_array_equal(out, expected)

        valid = chain.Valid[...].wait()
        np.testing.assert_array_equal(valid[0], 0)
        np.testing.assert_array_equal(valid[1:], 1)
