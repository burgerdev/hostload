
import numpy as np

from sklearn.svm import SVC
from sklearn.svm import SVR


from .abcs import OpTrain
from .abcs import OpPredict

from deeplearning.tools import SubRegion
from deeplearning.tools import Classification
from deeplearning.tools import Regression


class OpSVMTrain(OpTrain, Regression, Classification):
    def execute(self, slot, subindex, roi, result):
        assert len(self.Train) == 2, "need data and target"
        assert len(self.Valid) == 2, "need data and target"
        assert roi.start[0] == 0
        assert roi.stop[0] == 1

        train = self.Train[0][...].wait().view(np.ndarray)
        valid = self.Valid[0][...].wait().view(np.ndarray)
        X = np.concatenate((train, valid), axis=0)

        train = self.Train[1][...].wait().view(np.ndarray)
        valid = self.Valid[1][...].wait().view(np.ndarray)
        y = np.concatenate((train, valid), axis=0)

        # redefinition of model in else-clause is ok
        # pylint: disable=R0204
        if len(y.shape) == 1 or y.shape[1] == 1:
            # use regression
            y = y.squeeze()
            model = SVR()
        else:
            # use classification
            y = np.argmax(y, axis=1)
            model = SVC()

        model.fit(X, y)
        result[0] = model
        # pylint: enable=R0204


class OpSVMPredict(OpPredict, Regression, Classification):
    def execute(self, slot, subindex, roi, result):
        a = roi.start[0]
        b = roi.stop[0]
        c = 0
        d = self.Input.meta.shape[1]

        new_roi = SubRegion(self.Input, start=(a, c), stop=(b, d))
        X = self.Input.get(new_roi).wait()

        model = self.Classifier[...].wait()[0]

        if isinstance(model, SVR):
            result[:, 0] = model.predict(X)
        elif isinstance(model, SVC):
            classes = model.predict(X)
            for i, c in enumerate(range(roi.start[1], roi.stop[1])):
                result[:, i] = classes == c
        else:
            raise ValueError("incompatible model '{}'".format(type(model)))
