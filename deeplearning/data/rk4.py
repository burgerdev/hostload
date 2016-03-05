"""
Runge-Kutta method and Mackey-Glass series
"""

import numpy as np


# pylint: disable=R0913
def time_delay_runge_kutta_4(fun, t_0, y_0, tau, history=None, steps=1000,
                             width=1):
    """
    apply the classic Runge Kutta method to a time delay differential equation

    f: t, y(t), y(t-tau) -> y'(t)
    """
    width = float(width)
    if not isinstance(y_0, np.ndarray):
        y_0 = np.ones((1,), dtype=np.float)*y_0
    dim = len(y_0)

    hist_steps = np.floor(tau/width)
    assert tau/width == hist_steps, "tau must be a multiple of width"
    hist_steps = int(hist_steps)

    if history is None:
        history = np.zeros((hist_steps, dim), dtype=np.float)
    else:
        assert len(history) == hist_steps

    fun_eval = np.zeros((steps+1+hist_steps, dim), dtype=y_0.dtype)
    fun_eval[:hist_steps] = history
    fun_eval[hist_steps] = y_0

    for step in range(steps):
        k_1 = fun(t_0, y_0, fun_eval[step])
        k_2 = fun(t_0 + width/2, y_0 + width/2*k_1, fun_eval[step])
        k_3 = fun(t_0 + width/2, y_0 + width/2*k_2, fun_eval[step])
        k_4 = fun(t_0 + width, y_0 + width*k_3, fun_eval[step])
        t_0 += width
        y_0 += width*(k_1 + 2*k_2 + 2*k_3 + k_4)/6
        fun_eval[step+1+hist_steps] = y_0
    return fun_eval[hist_steps:]


# mackey glass does not depend on time explicitly, disable "unused arg"
# pylint: disable=W0613
# difference equations just need that many arguments, deal with it
# pylint: disable=R0913
def mackey_glass(beta, gamma, tau, exponent, steps=1000, width=.1):
    """
    approximation for Mackey-Glass-Function

    definition from http://www.hindawi.com/journals/ddns/2014/193758/
    """
    def mg_fun(time, fun, fun_delay):
        """
        Mackey-Glass time series depends only on current and delayed value
        """
        return beta*fun_delay/(1+fun_delay**exponent) + gamma*fun

    return time_delay_runge_kutta_4(mg_fun, 0, 1.2, tau, steps=steps,
                                    width=width)


def default_mackey_glass_series():
    """
    get an interesting MG-series for learning
    """
    return mackey_glass(.2, -.1, 17, 10, steps=12000, width=.1)


# uncomment to visualize Mackey-Glass series
'''
import vigra
from matplotlib import pyplot as plt

from deeplearning.targets import OpExponentiallySegmentedPattern
from deeplearning.tools import Graph


def main():
    plt.figure()
    mg_series = default_mackey_glass_series()
    timesteps = np.arange(len(mg_series))
    plt.plot(timesteps, mg_series)
    plt.hold(True)

    mg_series = vigra.taggedView(mg_series, axistags='tc').withAxes('t')

    mean = OpExponentiallySegmentedPattern(graph=Graph())
    mean.Input.setValue(mg_series)
    mean.NumSegments.setValue(1)

    legend = ["MackeyGlass"]
    for num in (8, 32, 128):
        mean.BaselineSize.setValue(num)
        smoothed = mean.Output[:].wait().squeeze()
        plt.plot(timesteps, smoothed)
        legend.append("mean over {}".format(num))

    plt.legend(legend)
    plt.show()


if __name__ == "__main__":
    main()
'''
