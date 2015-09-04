
import tempfile
import os

from lazyflow.graph import Graph


class Workflow(object):
    @staticmethod
    def build(d, workingdir=None):
        assert "class" in d and issubclass(Workflow, d["class"])

        if workingdir is None:
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
            os.mkdir(subdir)
            kwargs["workingdir"] = subdir
            setattr(w, attr, d[key]["class"].build(d[key], **kwargs))

        w._initialize()

        return w

    def __init__(self, workingdir=None):
        self._graph = Graph()
        self._workingdir = workingdir

    def run(self):
        self._predictionCache.Output[...].block()

    def _initialize(self):
        source = self._source
        features = self._features
        target = self._target
        split = self._split
        train = self._train
        cc = self._classifierCache
        predict = self._predict
        pc = self._predictionCache

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

        pc.Input.connect(predict.Output)