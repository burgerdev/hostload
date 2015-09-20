
import numpy as np

from .abcs import OpTrain
from .abcs import OpPredict

from lazyflow.classifiers import VigraRfLazyflowClassifierFactory
from lazyflow.classifiers import VigraRfLazyflowClassifier
from lazyflow.rtype import SubRegion

class OpRFTrain(OpTrain):
    @classmethod
    def build(cls, d, parent=None, graph=None, workingdir=None):
        op = cls(parent=parent, graph=graph)
        return op

    def execute(self, slot, subindex, roi, result):
        assert len(self.Train) == 2, "need data and target"
        assert len(self.Valid) == 2, "need data and target"
        assert roi.start[0] == 0
        assert roi.stop[0] == 1

        train = self.Train[0][...].wait()
        valid = self.Valid[0][...].wait()
        X = np.concatenate((train, valid), axis=0)

        assert len(self.Train[1].meta.shape) == 2,\
            "target needs to be a matrix"
        assert len(self.Valid[1].meta.shape) == 2,\
            "target needs to be a matrix"
        train = self.Train[1][...].wait()
        valid = self.Valid[1][...].wait()
        y = np.concatenate((train, valid), axis=0)

        y = np.argmax(y, axis=1)

        factory = VigraRfLazyflowClassifierFactory()
        classifier = factory.create_and_train(X, y)
        result[0] = classifier


class OpRFPredict(OpPredict):
    def execute(self, slot, subindex, roi, result):
        a = roi.start[0]
        b = roi.stop[0]
        c = 0
        d = self.Input.meta.shape[1]

        new_roi = SubRegion(self.Input, start=(a, c), stop=(b, d))
        X = self.Input.get(new_roi).wait()

        classifier = self.Classifier[...].wait()[0]
        assert isinstance(classifier, VigraRfLazyflowClassifier),\
            "type was {}".format(type(classifier))

        probs = classifier.predict_probabilities(X)
        result[...] = probs[:, roi.start[1]:roi.stop[1]]
