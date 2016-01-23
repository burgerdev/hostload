"""
Windowed operators that can be expressed as convolutions
"""

import numpy as np

from .window import OpWindow


class OpFilter(OpWindow):
    """
    base class for filter operations

    subclasses provide the get_filter() method
    """
    @classmethod
    def apply_window_function(cls, input_array, window_size, output_array):
        filter_ = cls.get_filter(window_size)
        output_array[:] = np.convolve(input_array, filter_, mode='valid')

    @classmethod
    def get_filter(cls, window_size):
        """
        return a filter of shape (window_size,)
        """
        raise NotImplementedError()


class OpMean(OpFilter):
    """
    mean value over window size

    m_k = \frac{1}{w}\sum_{i=1}^w x_{k-i+1}
    """
    @classmethod
    def get_filter(cls, window):
        return np.ones((window,), dtype=np.float32)/window


class OpLinearWeightedMean(OpFilter):
    """
    linear weighted mean over window size

    m_k = \frac{2}{w(w+1)}\sum_{i=1}^w \frac{x_{k-i+1}}{i}

    """
    @classmethod
    def get_filter(cls, window):
        filter_ = np.arange(window, dtype=np.float32)
        filter_ = window - filter_
        filter_ *= 2.0/(window*(window+1))
        return filter_


class OpExponentialFilter(OpFilter):
    """
    exponential filter

    parameter \lambda is chosen such that 99% of weight is inside filter
    """
    @classmethod
    def get_filter(cls, window):
        # F(x) = 1 - exp(-lambda*x) >= .99
        # lambda >= -log(0.01)/x
        lambda_ = -np.log(0.01)/window
        filter_ = np.arange(window, dtype=np.float32)
        filter_ = lambda_ * np.exp(-lambda_ * filter_)
        filter_ /= filter_.sum()
        return filter_


class OpGaussianSmoothing(OpFilter):
    """
    gaussian smoothing with a radius of (window - 1) / 2
    """
    @classmethod
    def get_filter(cls, window):
        assert window % 2 == 1,\
            "window size for gaussian kernel must be odd"
        radius = (window - 1) / 2.0
        sigma = radius / 3
        radius_range = np.linspace(-radius, radius, window)
        filter_ = np.exp(-radius_range**2/(2*sigma**2))
        filter_ /= filter_.sum()
        return filter_
