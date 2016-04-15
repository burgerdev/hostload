# -*- coding: utf-8 -*-

import numpy as np

from matplotlib import pyplot as plt

from deeplearning.data.rk4 import default_mackey_glass_series

if __name__ == "__main__":
    data = default_mackey_glass_series()
    upper = data.max()
    lower = data.min()
    data = (data - lower)/(upper - lower)
    data = data.squeeze()

    start = 4000
    stop = 8000

    time = np.arange(len(data))

    data = data[start:stop]
    time = time[start:stop]

    plt.plot(time, data)
    plt.xlabel("time step")
    plt.ylabel("Mackey-Glass series")
    plt.show()
