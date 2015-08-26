
import numpy as np

from opWindow import OpWindow


class OpFairness(OpWindow):
    @classmethod
    def applyWindowFunction(cls, input_array, window_size, output_array):
        sum_filter = np.ones((window_size,), dtype=np.float32)/window_size
        squares = np.square(input_array)
        sum_of_squares = np.convolve(squares, sum_filter, mode='valid')
        sums = np.convolve(input_array, sum_filter, mode='valid')
        squares_of_sums = np.square(sums)
        output_array[:] = squares_of_sums/sum_of_squares
