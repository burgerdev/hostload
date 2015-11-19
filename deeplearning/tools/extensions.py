
import os
import cPickle as pkl

from pylearn2.train_extensions import TrainExtension
from deeplearning.tools import Buildable


class BuildableTrainExtension(TrainExtension, Buildable):
    @classmethod
    def build(cls, config, workingdir=None):
        return cls(workingdir)

    def __init__(self, workingdir):
        self._wd = workingdir
        super(BuildableTrainExtension, self).__init__()


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

    def on_monitor(self, model, dataset, algorithm):
        """
        save the model's weights
        """
        channel = algorithm.monitor.channels["valid_objective"]
        prog = channel.val_shared.get_value()
        self._progress.append(prog)

    def setup(self, model, dataset, algorithm):
        """
        initialize the weight list
        """
        self._progress = []

    def get_progress(self):
        return self._progress

    def store(self):
        path = os.path.join(self._wd, "progressmonitor.pkl")
        with open(path, "w") as f:
            pkl.dump(self._progress, f)
