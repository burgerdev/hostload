
import tempfile
import os

from lazyflow.graph import Graph

from deeplearning.tools.serialization import dumps
from deeplearning.tools import Classification
from deeplearning.tools import Regression
from deeplearning.tools import IncompatibleTargets


class Workflow(object):
    @staticmethod
    def build(d, workingdir=None):
        assert "class" in d and issubclass(Workflow, d["class"])

        # we might delete keys, use a copy so that input is not changed
        d = d.copy()

        if workingdir is None:
            if "workingdir" in d:
                workingdir = d["workingdir"]
                del d["workingdir"]
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
        w = Workflow(workingdir=workingdir)

        kwargs = dict(graph=w._graph)

        for key in d:
            if key == "class":
                continue
            assert isinstance(key, str)
            attr = "_" + key
            assert not hasattr(w, attr)

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
            subdict = d[key]
            setattr(w, attr, subdict["class"].build(subdict, **kwargs))

        w._initialize()

        w._writeConfig(d)

        try:
            w._sanity_check()
        except IncompatibleTargets:
            w._cleanup()
            raise

        return w

    def __init__(self, workingdir=None):
        self._graph = Graph()
        self._workingdir = workingdir

    def run(self):
        self._report.Output[...].block()

    def _initialize(self):
        source = self._source
        features = self._features
        target = self._target
        split = self._split
        train = self._train
        cc = self._classifierCache
        predict = self._predict
        pc = self._predictionCache
        report = self._report

        features.Input.connect(source.Output)
        target.Input.connect(source.Output)

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
        report.Description.connect(split.Description)

    def _cleanup(self):
        print("cleaning")
        c = self._predictionCache
        self._report.All[0].disconnect()
        c.Input.disconnect()
        del self._predictionCache
        del c

    def _sanity_check(self):
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

    def _writeConfig(self, d):
        s = dumps(d, indent=4, sort_keys=True)
        fn = os.path.join(self._workingdir, "config.json")
        with open(fn, "w") as f:
            f.write(s)
            f.write("\n")
