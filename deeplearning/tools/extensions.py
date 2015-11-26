
import os
import cPickle as pkl

from itertools import imap

from pylearn2.train_extensions import TrainExtension
from deeplearning.tools import Buildable


class BuildableTrainExtension(TrainExtension, Buildable):
    @classmethod
    def build(cls, config, parent=None, graph=None, workingdir=None):
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

try:
    from pympler import summary
    from pympler import muppy
    from pympler import tracker
except ImportError:
    pass
else:
    class MemoryDebugger(PersistentTrainExtension):
        """
        uses the pympler module for debugging memory leaks
        """

        def on_monitor(self, model, dataset, algorithm):
            """
            add a new report about leaked objects
            """
            current_summary = summary.summarize(muppy.get_objects())
            self._diffs.append(self._tr.format_diff(self._last_summary,
                                                    current_summary))

            self._last_summary = current_summary

        def setup(self, model, dataset, algorithm):
            """
            initialize the weight list
            """
            self._diffs = []
            self._tr = tracker.SummaryTracker()
            self._last_summary = summary.summarize(muppy.get_objects())

        def store(self):
            path = os.path.join(self._wd, "memleaks.log")
            with open(path, "w") as f:
                for gen in self._diffs:
                    lines = imap("{}\n".format, gen)
                    f.writelines(lines)
                f.write("=============== EPOCH ===============\n")
