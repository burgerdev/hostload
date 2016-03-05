
import os
import cPickle as pkl

import numpy as np

from pylearn2.train_extensions import TrainExtension

from .abcs import Buildable


class BuildableTrainExtension(TrainExtension, Buildable):
    @classmethod
    def build(cls, config, parent=None, graph=None, workingdir=None):
        """
        build an instance of this class with given configuration dict
        """
        config_copy = config.copy()
        if "wd" not in config_copy:
            config_copy["wd"] = workingdir
        obj = super(BuildableTrainExtension, cls).build(config_copy)

        return obj

    def __init__(self, **kwargs):
        if "workingdir" in kwargs:
            self._wd = kwargs["workingdir"]
        super(BuildableTrainExtension, self).__init__()

    @classmethod
    def get_default_config(cls):
        """
        override to provide your own default configuration
        """
        conf = super(BuildableTrainExtension, cls).get_default_config()
        conf["wd"] = None
        return conf


class PersistentTrainExtension(BuildableTrainExtension):
    def store(self):
        """
        store the findings of this extension
        """
        pass


class WeightKeeper(PersistentTrainExtension):
    """
    keeps track of the model's weights at each monitor step
    """
    _weights = []

    def on_monitor(self, model, dataset, algorithm):
        """
        save the model's weights
        """
        self._weights.append(model.get_param_values())

    def setup(self, model, dataset, algorithm):
        """
        initialize the weight list
        """
        self._weights = []

    def get_weights(self):
        return self._weights

    def store(self):
        path = os.path.join(self._wd, "weightkeeper.pkl")
        with open(path, "w") as f:
            pkl.dump(self._weights, f)


class ProgressMonitor(PersistentTrainExtension):
    """
    keeps track of the model's weights at each monitor step
    """

    _progress = np.NaN

    @classmethod
    def get_default_config(cls):
        config = super(ProgressMonitor, cls).get_default_config()
        config["channel"] = "valid_objective"
        return config

    def on_monitor(self, model, dataset, algorithm):
        """
        save the desired channel
        """
        monitor = model.monitor
        channels = monitor.channels
        channel = channels[self._channel]
        self._progress = channel.val_record

    def get_progress(self):
        return self._progress

    def store(self):
        filename = "progress_{}.pkl".format(self._channel)
        path = os.path.join(self._wd, filename)
        with open(path, "w") as f:
            pkl.dump(self._progress, f)


class MonitorBasedSaveBest(BuildableTrainExtension):
    """
    similar to pylearn2's MonitorBasedSaveBest, but avoids memory hogging
    (see https://github.com/lisa-lab/pylearn2/issues/1567)
    """

    best_cost = np.inf
    best_params = None

    @classmethod
    def get_default_config(cls):
        config = super(MonitorBasedSaveBest, cls).get_default_config()
        config["channel"] = "valid_objective"
        return config

    def setup(self, model, dataset, algorithm):
        self.best_cost = np.inf
        self.best_params = model.get_param_values()

    def on_monitor(self, model, dataset, algorithm):
        """
        Looks whether the model performs better than earlier. If it's the
        case, saves the model.

        Parameters
        ----------
        model : pylearn2.models.model.Model
            model.monitor must contain a channel with name given by
            self.channel_name
        dataset : pylearn2.datasets.dataset.Dataset
            Not used
        algorithm : TrainingAlgorithm
            Not used
        """
        monitor = model.monitor
        channels = monitor.channels
        channel = channels[self._channel]
        val_record = channel.val_record
        new_cost = val_record[-1]

        if new_cost < self.best_cost:
            self.best_cost = new_cost
            self.best_params = model.get_param_values()


try:
    from .mem import traverse
    from .mem import get_arrays
except ImportError:
    pass
else:
    class MemoryDebugger(PersistentTrainExtension):
        """
        uses the pympler module for debugging memory leaks
        """
        _new_arrays = []
        _id_set = set()

        def on_monitor(self, model, dataset, algorithm):
            """
            add a new report about leaked objects
            """
            self._update_arrays()

        def setup(self, model, dataset, algorithm):
            """
            initialize the weight list
            """
            self._new_arrays = []
            self._id_set = set()
            self._update_arrays()

        def store(self):
            traverse(self._new_arrays, start=True)

        def _update_arrays(self):
            self._new_arrays = [a for a in get_arrays()
                                if id(a) not in self._id_set]
            self._id_set.update(set([id(a) for a in self._new_arrays]))
