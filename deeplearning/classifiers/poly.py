
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from lazyflow.rtype import SubRegion
from lazyflow.operator import InputSlot

from .abcs import OpTrain
from .abcs import OpPredict

from deeplearning.tools import Regression


class OpPolynomialTrain(OpTrain, Regression):
    Degree = InputSlot(optional=True)
    __degree = 3

    def setupOutputs(self):
        super(OpPolynomialTrain, self).setupOutputs()
        if self.Degree.ready():
            self.__degree = self.Degree.value
        else:
            # single underscore is from config, double is used later on
            self.__degree = self._degree

    @classmethod
    def get_default_config(cls):
        conf = super(OpPolynomialTrain, cls).get_default_config()
        conf["degree"] = cls.__degree
        return conf

    def execute(self, slot, subindex, roi, result):
        assert len(self.Train) == 2, "need data and target"
        assert len(self.Valid) == 2, "need data and target"
        assert roi.start[0] == 0
        assert roi.stop[0] == 1

        train = self.Train[0][...].wait().view(np.ndarray)
        valid = self.Valid[0][...].wait().view(np.ndarray)
        features = np.concatenate((train, valid), axis=0)

        train = self.Train[1][...].wait().view(np.ndarray)
        valid = self.Valid[1][...].wait().view(np.ndarray)
        target = np.concatenate((train, valid), axis=0)
        target = target.squeeze()

        poly = PolynomialFeatures(degree=self.__degree)
        # polynomial features contain column of 1's, no need for fit_intercept
        linear = LinearRegression(fit_intercept=False)
        model = Pipeline([("poly", poly), ("linear", linear)])

        model.fit(features, target)
        result[0] = model


class OpPolynomialPredict(OpPredict, Regression):
    def execute(self, slot, subindex, roi, result):
        start_t = roi.start[0]
        stop_t = roi.stop[0]
        start_c = 0
        stop_c = self.Input.meta.shape[1]

        new_roi = SubRegion(self.Input, start=(start_t, start_c),
                            stop=(stop_t, stop_c))
        features = self.Input.get(new_roi).wait()

        model = self.Classifier[...].wait()[0]

        if not isinstance(model, Pipeline):
            raise ValueError("unsupported model '{}'".format(type(model)))

        result[:, 0] = model.predict(features)
