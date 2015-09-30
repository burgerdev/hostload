
from functools import wraps
import logging

import numpy as np

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion

from pylearn2.datasets import Dataset
from pylearn2.space import CompositeSpace
from pylearn2.utils.iteration import ShuffledSequentialSubsetIterator


logger = logging.getLogger(__name__)


def _assert_input_ready(method):
    """
    wrapper for OpDataset methods to prevent usage before input is ready
    """
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        assert len(self.Input) == 2, "input slot needs data and target"
        assert self.Input[0].ready() and self.Input[1].ready(),\
            "input is not ready, can't use dataset yet"
        return method(self, *args, **kwargs)
    return wrapped


def _warn_if_unwise(method):
    """
    wrapper for OpDataset methods to prevent usage before input is ready
    """
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        n = self.Input[0].meta.shape[0]
        c = self.Input[0].meta.shape[0]
        if n*c*4 > 1024**3:
            msg = "requested non-lazy processing for large dataset"
            logger.warn(msg)
        return method(self, *args, **kwargs)
    return wrapped


class OpDataset(Operator, Dataset):
    """
    converts an input slot to a pylearn2 dataset

    the Output slot simply replicates the input slot, accessing data is done
    via the pylearn2.dataset.Dataset interface
    """
    Input = InputSlot(level=1)
    Output = OutputSlot(level=1)

    slotNames = dict(features=0, targets=1)

    def __init__(self, *args, **kwargs):
        Operator.__init__(self, *args, **kwargs)
        self.Output.connect(self.Input)

    def setupOutputs(self):
        self._num_examples = self.Input[0].meta.shape[0]
        assert self._num_examples == self.Input[1].meta.shape[0]
        # self._setupDataset()

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(roi)

    def execute(self, slot, subindex, roi, result):
        raise NotImplementedError("should not reach this method")

    # METHODS FOR DATASET

    def has_targets(self):
        return True

    @_assert_input_ready
    def get_num_examples(self):
        return self._num_examples

    @_assert_input_ready
    def iterator(self, mode=None, batch_size=None, num_batches=None, rng=None,
                 data_specs=None, return_tuple=False):
        if mode is None:
            mode = "sequential"

        supported_modes = {
            "sequential": self._getSequentialIterator,
            "shuffled_sequential": self._getShuffledSequentialIterator}

        if mode not in supported_modes:
            msg = "mode '{}' not implemented".format(mode)
            mode = "sequential"
            msg += ", defaulting to '{}'".format(mode)
            logger.warn(msg)

        return supported_modes[mode](batch_size=batch_size,
                                     num_batches=num_batches,
                                     rng=rng, data_specs=data_specs,
                                     return_tuple=return_tuple)

    def _getSequentialIterator(self, batch_size=None, num_batches=None,
                               rng=None, data_specs=None,
                               return_tuple=False):

        if rng is not None:
            logger.warn("rng handling not implemented")

        n = int(self._num_examples)

        if batch_size is None:
            if num_batches is None:
                batch_size = n
                num_batches = 1
            else:
                batch_size = int(np.ceil(float(n)/num_batches))
        else:
            num_batches = int(np.ceil(float(n)/batch_size))

        data_types = self._getDataTypes(data_specs)

        shapes = [tuple(s.meta.shape)[1:] for s in self.Input]

        def _iter():
            for b in range(num_batches):
                left = b*batch_size
                right = (b+1)*batch_size
                right = min(right, n)

                ret = []

                for data_type in data_types:
                    s = shapes[data_type]
                    start = (left,) + (0,)*len(s)
                    stop = (right,) + s
                    new_roi = SubRegion(self.Input, start=start, stop=stop)
                    X = self.Input[data_type].get(new_roi).wait()
                    X = X.astype(np.float32)
                    ret.append(X)

                if return_tuple or len(ret) > 1:
                    yield tuple(ret)
                else:
                    assert len(ret) == 1
                    yield ret[0]

        return _Iterator(batch_size, num_batches, self._num_examples,
                         _iter())

    @_warn_if_unwise
    def _getShuffledSequentialIterator(self, batch_size=None,
                                       num_batches=None,
                                       rng=None, data_specs=None,
                                       return_tuple=False):

        X = self.Input[0][...].wait()
        y = self.Input[1][...].wait()
        data = (X, y)

        index_iter = ShuffledSequentialSubsetIterator(
            len(X), batch_size, num_batches, rng=rng)

        data_types = self._getDataTypes(data_specs)

        def _iter():
            for indices in index_iter:

                ret = []

                for data_type in data_types:
                    temp = data[data_type][indices, ...]
                    temp = temp.astype(np.float32)
                    ret.append(temp)

                if return_tuple or len(ret) > 1:
                    yield tuple(ret)
                else:
                    assert len(ret) == 1
                    yield ret[0]

        it = _Iterator(batch_size, num_batches, self._num_examples,
                       _iter())
        it.stochastic = True
        return it

    def _getDataTypes(self, data_specs):
        # default data returned is 'features'
        data_types = (0,)
        if data_specs is not None:
            assert len(data_specs) == 2
            space = data_specs[0]
            if isinstance(space, CompositeSpace):
                data_types = map(lambda k: self.slotNames[k],
                                 data_specs[1])
            else:
                data_types = (self.slotNames[data_specs[1]],)

        return data_types


class _Iterator(object):

    def __init__(self, batch_size, num_batches, num_examples, it):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_examples = num_examples
        self.uneven = True
        self.fancy = False
        self.stochastic = False

        self._it = iter(it)

    def __iter__(self):
        return self

    def next(self):
        return self._it.next()
