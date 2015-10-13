# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:39:33 2015

@author: burger
"""

import numpy as np


def time_delay_runge_kutta_4(f, t_0, y_0, tau, history=None, n=1000, h=1):
    """
    apply the classic Runge Kutta method to a time delay differential equation

    f: t, y(t), y(t-tau) -> y'(t)
    """
    h = float(h)
    if not isinstance(y_0, np.ndarray):
        y_0 = np.ones((1,), dtype=np.float)*y_0
    d = len(y_0)

    hist_steps = np.floor(tau/h)
    assert tau/h == hist_steps, "tau must be a multiple of h"
    hist_steps = int(hist_steps)

    if history is None:
        history = np.zeros((hist_steps, d), dtype=np.float)
    else:
        assert len(history) == hist_steps

    y = np.zeros((n+1+hist_steps, d), dtype=y_0.dtype)
    y[:hist_steps] = history
    y[hist_steps] = y_0

    t_n = t_0
    y_n = y_0
    for step in range(n):
        y_delayed = y[step]
        k_1 = f(t_n, y_n, y_delayed)
        k_2 = f(t_n + h/2, y_n + h/2*k_1, y_delayed)
        k_3 = f(t_n + h/2, y_n + h/2*k_2, y_delayed)
        k_4 = f(t_n + h, y_n + h*k_3, y_delayed)
        t_n += h
        y_n += h*(k_1 + 2*k_2 + 2*k_3 + k_4)/6
        y[step+1+hist_steps] = y_n
    return y[hist_steps:]


def mackey_glass(beta, gamma, tau, exponent, n=1000, h=.1):
    """
    approximation for Mackey-Glass-Function

    definition from http://www.hindawi.com/journals/ddns/2014/193758/
    """
    def mg_fun(t, y, y_delay):
        return beta*y_delay/(1+y_delay**exponent) + gamma*y

    return time_delay_runge_kutta_4(mg_fun, 0, 1.2, tau, n=n, h=h)
        

def default_mackey_glass_series():
    """
    get an interesting MG-series for learning
    """
    return mackey_glass(.2, -.1, 17, 10, n=12000, h=.1)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from deeplearning.targets import OpExponentiallySegmentedPattern
    from lazyflow.graph import Graph
    import vigra

    plt.figure()
    y = default_mackey_glass_series()
    t = np.arange(len(y))
    plt.plot(t, y)
    plt.hold(True)

    y = vigra.taggedView(y, axistags='tc').withAxes('t')

    mean = OpExponentiallySegmentedPattern(graph=Graph())
    mean.Input.setValue(y)
    mean.NumSegments.setValue(1)

    legend = ["MackeyGlass"]
    for num in (8, 32, 128):
        mean.BaselineSize.setValue(num)
        z = mean.Output[:].wait().squeeze()
        plt.plot(t, z)
        legend.append("mean over {}".format(num))

    plt.legend(legend)
    plt.show()    
