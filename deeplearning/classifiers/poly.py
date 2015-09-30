
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from lazyflow.rtype import SubRegion
from lazyflow.operator import InputSlot

from .abcs import OpTrain
from .abcs import OpPredict

from deeplearning.tools import Classification
from deeplearning.tools import Regression


class OpPolynomialTrain(OpTrain, Regression):
    Degree = InputSlot()

    @classmethod
    def build(cls, d, parent=None, graph=None, workingdir=None):
            op = cls(parent=parent, graph=graph)
            my_d = {"degree": 3}
            my_d.update(d)
            op.Degree.setValue(my_d["degree"])

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
        y = y.squeeze()

        degree = self.Degree.value

        poly = PolynomialFeatures(degree=degree)
        # polynomial features contain column of 1's, no need for fit_intercept
        linear = LinearRegression(fit_intercept=False)
        model = Pipeline([("poly", poly), ("linear", linear)])

        model.fit(X, y)
        result[0] = model


class OpPolynomialPredict(OpPredict, Regression):
    def execute(self, slot, subindex, roi, result):
        a = roi.start[0]
        b = roi.stop[0]
        c = 0
        d = self.Input.meta.shape[1]

        new_roi = SubRegion(self.Input, start=(a, c), stop=(b, d))
        X = self.Input.get(new_roi).wait()

        model = self.Classifier[...].wait()[0]

        if not isinstance(model, Pipeline):
            raise ValueError("unsupported model '{}'".format(type(model)))

        result[:, 0] = model.predict(X)
