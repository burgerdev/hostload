
import numpy as np

from opWindow import OpWindow

class OpFilter(OpWindow):
    @classmethod
    def applyWindowFunction(cls, input_array, window_size, output_array):
        f = cls.getFilter(window_size)
        output_array[:] = np.convolve(input_array, f, mode='valid')

    @classmethod
    def getFilter(cls, window_size):
        raise NotImplementedError()


class OpMean(OpFilter):
    @classmethod
    def getFilter(cls, window):
        return np.ones((window,), dtype=np.float32)/window


class OpLinearWeightedMean(OpFilter):
    @classmethod
    def getFilter(cls, window):
        f = np.arange(window, dtype=np.float32)
        f = window - f
        f *= 2.0/(window*(window+1))
        return f
