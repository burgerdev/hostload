
import numpy as np

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion

from pylearn2.datasets import Dataset


class OpDataset(Operator, Dataset):
    """
    converts an input slot to a pylearn2 dataset

    the Output slot simply replicates the input slot, accessing data is done
    via the pylearn2.dataset.Dataset interface
    """
    Input = InputSlot()
    Output = OutputSlot()

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)
        self._num_examples = self.Input.meta.shape[0]
        self._example_shape = tuple(self.Input.meta.shape)[1:]

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(roi)

    def execute(self, slot, subindex, roi, result):
        req = self.Input.get(roi)
        req.writeInto(result)
        req.block()

    # METHODS FOR DATASET

    def has_targets(self):
        return False

    def get_num_examples(self):
        return self._num_examples

    def iterator(self, mode=None, batch_size=None, num_batches=None, rng=None,
                 data_specs=None, return_tuple=False):
        if mode is None:
            mode = "sequential"

        assert mode == "sequential", "{} not implemented".format(mode)
        assert rng is None, "rng handling not implemented"

        n = int(self._num_examples)
        s = self._example_shape

        if batch_size is None:
            if num_batches is None:
                batch_size = n
                num_batches = 1
            else:
                batch_size = int(np.ceil(float(n)/num_batches))
        else:
            num_batches = int(np.ceil(float(n)/batch_size))

        for b in range(num_batches):
            left = b*batch_size
            right = (b+1)*batch_size
            right = min(right, n)

            start = (left,) + (0,)*len(s)
            stop = (right,) + s
            new_roi = SubRegion(self.Input, start=start, stop=stop)
            X = self.Input.get(new_roi).wait()

            if return_tuple:
                yield (X,)
            else:
                yield X
