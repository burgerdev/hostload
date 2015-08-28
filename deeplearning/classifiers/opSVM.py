
import numpy as np

from sklearn.svm import SVC

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion
from lazyflow.stype import Opaque


class OpSVMTrain(Operator):
    Train = InputSlot(level=1)
    Valid = InputSlot(level=1)

    Classifier = OutputSlot()

    def setupOutputs(self):
        self.Classifier.meta.shape = (1,)
        self.Classifier.meta.dtype = np.object

    def propagateDirty(self, slot, subindex, roi):
        self.Classifier.setDirty(slice(None))

    def execute(self, slot, subindex, roi, result):
        assert len(self.Train) == 2, "need data and target"
        assert len(self.Valid) == 2, "need data and target"
        assert roi.start[0] == 0
        assert roi.stop[0] == 1

        train = self.Train[0][...].wait()
        valid = self.Valid[0][...].wait()
        X = np.concatenate((train, valid), axis=0)

        assert len(self.Train[1].meta.shape) == 1,\
            "target needs to be a vector"
        assert len(self.Valid[1].meta.shape) == 1,\
            "target needs to be a vector"
        train = self.Train[1][...].wait()
        valid = self.Valid[1][...].wait()
        y = np.concatenate((train, valid), axis=0)

        svc = SVC()
        svc.fit(X, y)

        result[0] = svc


class OpSVMPredict(Operator):
    Input = InputSlot()
    Classifier = InputSlot()

    Output = OutputSlot()

    def setupOutputs(self):
        n = self.Input.meta.shape[0]
        self.Output.meta.shape = (n,)
        self.Output.meta.dtype = np.float32

    def propagateDirty(self, slot, subindex, roi):
        if slot == self.Classifier:
            self.Output.setDirty(slice(None))
        else:
            a = roi.start[0]
            b = roi.stop[0]
            new_roi = SubRegion(self.Output, start=(a,), stop=(b,))
            self.Output.setDirty(new_roi)

    def execute(self, slot, subindex, roi, result):
        a = roi.start[0]
        b = roi.stop[0]
        c = 0
        d = self.Input.meta.shape[1]

        new_roi = SubRegion(self.Input, start=(a, c), stop=(b, d))
        X = self.Input.get(new_roi).wait()

        svc = self.Classifier[...].wait()[0]
        assert isinstance(svc, SVC), "type was {}".format(type(svc))
        result[:] = svc.predict(X)
