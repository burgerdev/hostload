
import numpy as np

from sklearn.svm import SVC

from lazyflow.rtype import SubRegion

from .abcs import OpTrain
from .abcs import OpPredict


class OpSVMTrain(OpTrain):
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

        svc = SVC()
        svc.fit(X, y)

        result[0] = svc


class OpSVMPredict(OpPredict):
    def execute(self, slot, subindex, roi, result):
        a = roi.start[0]
        b = roi.stop[0]
        c = 0
        d = self.Input.meta.shape[1]

        new_roi = SubRegion(self.Input, start=(a, c), stop=(b, d))
        X = self.Input.get(new_roi).wait()

        svc = self.Classifier[...].wait()[0]
        assert isinstance(svc, SVC), "type was {}".format(type(svc))

        classes = svc.predict(X)
        for i, c in enumerate(range(roi.start[1], roi.stop[1])):
            result[:, i] = classes == c
