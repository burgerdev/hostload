"""
Operator to convert lazyflow slots into pylearn2 datasets.
"""

from functools import wraps
import logging

import numpy as np

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion

from pylearn2.datasets import Dataset
from pylearn2.space import CompositeSpace
from pylearn2.utils.iteration import SubsetIterator
from pylearn2.utils.iteration import ShuffledSequentialSubsetIterator


LOGGER = logging.getLogger(__name__)


# docstring is applied by @wraps
# pylint: disable-msg=C0111
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
    warns if a method is going to have low performance on a lazyflow dataset
    """
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        num_examples = self.Input[0].meta.shape[0]
        num_channels = self.Input[0].meta.shape[0]
        if num_examples*num_channels*4 > 1024**3:
            msg = "requested non-lazy processing for large dataset"
            LOGGER.warn(msg)
        return method(self, *args, **kwargs)
    return wrapped


# pylint: enable-msg=C0111
class OpDataset(Operator, Dataset):
    """
    converts an input slot to a pylearn2 dataset

    the Output slot simply replicates the input slot, accessing data is done
    via the pylearn2.dataset.Dataset interface
    """
    Input = InputSlot(level=1)
    Output = OutputSlot(level=1)

    slotNames = dict(features=0, targets=1)

    _num_examples = -1

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

    # we need all those arguments
    # pylint: disable-msg=R0913
    @_assert_input_ready
    def iterator(self, mode=None, batch_size=None, num_batches=None, rng=None,
                 data_specs=None, return_tuple=False):
        if mode is None:
            mode = "sequential"

        supported_modes = {
            "sequential": self._get_sequential_iterator,
            "shuffled_sequential": self._get_shuffled_iterator}

        if mode not in supported_modes:
            msg = "mode '{}' not implemented".format(mode)
            mode = "sequential"
            msg += ", defaulting to '{}'".format(mode)
            LOGGER.warn(msg)

        return supported_modes[mode](batch_size=batch_size,
                                     num_batches=num_batches,
                                     rng=rng, data_specs=data_specs,
                                     return_tuple=return_tuple)

    def _get_sequential_iterator(self, batch_size=None, num_batches=None,
                                 rng=None, data_specs=None,
                                 return_tuple=False):
        """
        construct a pylearn2 SubsetIterator for this dataset specification
        """
        if rng is not None:
            LOGGER.warn("rng handling not implemented")

        num_examples = int(self._num_examples)

        if batch_size is None:
            if num_batches is None:
                batch_size = num_examples
                num_batches = 1
            else:
                batch_size = int(np.ceil(float(num_examples)/num_batches))
        else:
            num_batches = int(np.ceil(float(num_examples)/batch_size))

        data_types = self._get_data_types(data_specs)

        shapes = [tuple(s.meta.shape)[1:] for s in self.Input]

        def _iter():
            """
            internal iterator, to be used by _Iterator
            """
            for batch in range(num_batches):
                left = batch*batch_size
                right = (batch+1)*batch_size
                right = min(right, num_examples)

                ret = []

                for data_type in data_types:
                    shape = shapes[data_type]
                    start = (left,) + (0,)*len(shape)
                    stop = (right,) + shape
                    new_roi = SubRegion(self.Input, start=start, stop=stop)
                    batch = self.Input[data_type].get(new_roi).wait()
                    # theano needs float32
                    batch = batch.astype(np.float32)
                    ret.append(batch)

                if return_tuple or len(ret) > 1:
                    yield tuple(ret)
                else:
                    assert len(ret) == 1
                    yield ret[0]

        return _Iterator(batch_size, num_batches, self._num_examples,
                         _iter())

    @_warn_if_unwise
    def _get_shuffled_iterator(self, batch_size=None, num_batches=None,
                               rng=None, data_specs=None, return_tuple=False):
        """
        iterate over dataset with randomly selected connected batches
        """
        features = self.Input[0][...].wait()
        target = self.Input[1][...].wait()
        data = (features, target)

        index_iter = ShuffledSequentialSubsetIterator(
            len(features), batch_size, num_batches, rng=rng)

        data_types = self._get_data_types(data_specs)

        def _iter():
            """
            internal iterator for _Iterator
            """
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

        iter_ = _Iterator(batch_size, num_batches, self._num_examples,
                          _iter())
        iter_.stochastic = True
        return iter_

    def _get_data_types(self, data_specs):
        """
        get a mapping of type to channel index
        """
        # default data returned is 'features'
        data_types = (0,)
        if data_specs is not None:
            assert len(data_specs) == 2
            space = data_specs[0]
            if isinstance(space, CompositeSpace):
                data_types = [self.slotNames[k] for k in data_specs[1]]
            else:
                data_types = (self.slotNames[data_specs[1]],)

        return data_types


# we don't want to call super().__init__!
# pylint: disable-msg=W0231
class _Iterator(SubsetIterator):
    """
    wrapper around a python iterator over batches

    pylearn2 insists on using its custom iterator class rather than the python
    __iter__ pattern, play along.
    """
    def __init__(self, batch_size, num_batches, num_examples, it):
        self._batch_size = batch_size
        self._num_batches = num_batches
        self._num_examples = num_examples
        self._uneven = True

        self._it = iter(it)

    def __iter__(self):
        return self

    def next(self):
        return self._it.next()

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def uneven(self):
        return self._uneven
