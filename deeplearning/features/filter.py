
import numpy as np

from .window import OpWindow


class OpFilter(OpWindow):
    """
    base class for filter operations

    subclasses provide the getFilter() method
    """
    @classmethod
    def applyWindowFunction(cls, input_array, window_size, output_array):
        f = cls.getFilter(window_size)
        output_array[:] = np.convolve(input_array, f, mode='valid')

    @classmethod
    def getFilter(cls, window_size):
        raise NotImplementedError()


class OpMean(OpFilter):
    """
    mean value over window size

    m_k = \frac{1}{w}\sum_{i=1}^w x_{k-i+1}
    """
    @classmethod
    def getFilter(cls, window):
        return np.ones((window,), dtype=np.float32)/window


class OpLinearWeightedMean(OpFilter):
    """
    linear weighted mean over window size

    m_k = \frac{2}{w(w+1)}\sum_{i=1}^w \frac{x_{k-i+1}}{i}

    """
    @classmethod
    def getFilter(cls, window):
        f = np.arange(window, dtype=np.float32)
        f = window - f
        f *= 2.0/(window*(window+1))
        return f


class OpExponentialFilter(OpFilter):
    """
    exponential filter

    parameter \lambda is chosen such that 99% of weight is inside filter
    """
    @classmethod
    def getFilter(cls, window):
        # F(x) = 1 - exp(-lambda*x) >= .99
        # lambda >= -log(0.01)/x
        lambda_ = -np.log(0.01)/window
        f = np.arange(window, dtype=np.float32)
        f = lambda_ * np.exp(-lambda_ * f)
        f /= f.sum()
        return f
