"""
The Workflow classes join data source definition, feature extraction,
train/test split, training, testing and reporting in a single runnable object.
"""


import tempfile
import os
import datetime

from lazyflow.graph import Graph

from deeplearning.tools.serialization import dumps
from deeplearning.tools import Classification
from deeplearning.tools import Regression
from deeplearning.tools import IncompatibleTargets
from deeplearning.tools import Buildable
from deeplearning.tools import build_operator

from deeplearning.data.caches import OpPickleCache
from deeplearning.data.caches import OpHDF5Cache
from deeplearning.split import OpTrainTestSplit

from deeplearning.report import OpRegressionReport


class Workflow(Buildable):
    """
    machine learning workflow
    """
    Features = None
    Prediction = None
    Target = None

    @classmethod
    def build(cls, outer_config, parent=None, graph=None, workingdir=None):
        """
        custom workflow setup isntructions
        """
        assert parent is None
        assert graph is None

        config = cls.get_default_config()
        config.update(outer_config)
        assert "class" in config and issubclass(config["class"], Workflow)
        del config["class"]

        if workingdir is None:
            if "workingdir" in config:
                workingdir = config["workingdir"]
                del config["workingdir"]
                try:
                    os.mkdir(workingdir)
                except OSError as err:
                    if "exists" in str(err):
                        # that's fine
                        pass
                    else:
                        raise
            else:
                workingdir = tempfile.mkdtemp(prefix="deeplearning_")

        w = cls(workingdir=workingdir, config=config)

        return w

    def __init__(self, workingdir=None, config=None):
        self._config = None
        self._graph = Graph()
        self._workingdir = workingdir
        self._start_time = None

        kwargs = dict(graph=self._graph)

        if config is None:
            return

        for key in config:
            assert isinstance(key, str)
            attr = "_" + key
            assert not hasattr(self, attr)

            if key == "preprocessing":
                value = [build_operator(subdict, **kwargs)
                         for subdict in config[key]]
            else:
                subdir = os.path.join(workingdir, key)
                try:
                    os.mkdir(subdir)
                except OSError as err:
                    if "exists" in str(err):
                        # that's fine
                        pass
                    else:
                        raise

                kwargs["workingdir"] = subdir
                value = build_operator(config[key], **kwargs)

            setattr(self, attr, value)

        self._initialize()

        self._config = config

    def run(self):
        self._pre_run()
        self._report.Output[...].block()
        self._post_run()

    def set_classifier(self, classifier):
        """
        force workflow to use an external classifier

        This is useful for reloading a previously trained model.
        """
        self._predict.Classifier.disconnect()
        self._predict.Classifier.setValue(classifier)

    def _initialize(self):
        """
        set up operator connections
        """
        source = self._source
        features = self._features
        target = self._target
        split = self._split
        train = self._train
        cc = self._classifierCache
        predict = self._predict
        pc = self._predictionCache
        report = self._report

        lastOutput = source.Output

        for op in self._preprocessing:
            op.Input.connect(lastOutput)
            lastOutput = op.Output

        features.Input.connect(lastOutput)
        target.Input.connect(lastOutput)

        split.Input.resize(2)
        split.Input[0].connect(features.Output)
        split.Input[1].connect(target.Output)

        train.Train.connect(split.Train)
        train.Valid.connect(split.Valid)

        cc.Input.connect(train.Classifier)

        predict.Classifier.connect(cc.Output)
        predict.Input.connect(split.All[0])
        predict.Target.connect(split.Train[1])

        pc.Input.connect(predict.Output)

        report.All.resize(2)
        report.All[0].connect(pc.Output)
        report.All[1].connect(target.Output)

        report.Valid.resize(2)
        report.Valid[0].connect(features.Valid)
        report.Valid[1].connect(target.Valid)
        report.Description.connect(split.Description)

        self.Prediction = predict.Output
        self.Features = split.All[0]
        self.Target = target.Output

    def _pre_run(self):
        """
        prepare everything for the next run
        """
        assert self._config is not None,\
            "workflow not configured - did you run build()?"
        # write config file
        s = dumps(self._config, indent=4, sort_keys=True)
        fn = os.path.join(self._workingdir, "config.json")
        with open(fn, "w") as f:
            f.write(s)
            f.write("\n")

        # perform sanity checks, terminate early if incompatible
        try:
            self._sanity_check()
        except IncompatibleTargets:
            self._cleanup()
            raise

        # keep time for reporting
        self._start_time = datetime.datetime.now()

    def _post_run(self):
        """
        process results of last run
        """
        # keep time for reporting
        time_elapsed = datetime.datetime.now() - self._start_time
        # write config file
        fn = os.path.join(self._workingdir, "elapsed_time.txt")
        with open(fn, "w") as f:
            f.write(str(time_elapsed))
            f.write("\n")

    def _cleanup(self):
        """
        deallocate everything so that we don't leak memory
        """
        c = self._predictionCache
        self._report.All[0].disconnect()
        c.Input.disconnect()
        c.cleanUp()
        del self._predictionCache
        del c

    def _sanity_check(self):
        """
        check if operators are compatible
        """
        if isinstance(self._target, Classification):
            t = Classification
        elif isinstance(self._target, Regression):
            t = Regression
        else:
            raise IncompatibleTargets("unkown target type")

        if not isinstance(self._train, t):
            raise IncompatibleTargets("incompatible training type")
        if not isinstance(self._predict, t):
            raise IncompatibleTargets("incompatible prediction type")
        if not isinstance(self._report, t):
            raise IncompatibleTargets("incompatible report type")

    @classmethod
    def get_default_config(cls):
        config = {"class": Workflow,
                  "source": {"class": None},
                  "preprocessing": tuple(),
                  "features": {"class": None},
                  "target": {"class": None},
                  "split": {"class": OpTrainTestSplit},
                  "train": {"class": None},
                  "classifierCache": {"class": OpPickleCache},
                  "predict": {"class": None},
                  "predictionCache": {"class": OpHDF5Cache},
                  "report": {"class": None}}
        return config


class RegressionWorkflow(Workflow):
    """
    workflow specialization for regression, provides default config entries
    """
    @classmethod
    def get_default_config(cls):
        config = super(RegressionWorkflow, cls).get_default_config()
        config["class"] = RegressionWorkflow
        config["report"] = {"class": OpRegressionReport, "levels": 50}
        return config