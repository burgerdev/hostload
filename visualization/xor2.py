# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 22:48:37 2016

@author: burger
"""

import numpy as np
from matplotlib import pyplot as plt


def sigma(x, a=1, b=0):
    return 1/(1+np.exp(-(a*x+b)))


x = np.asarray([[0.0, .1], [0, 1], [.9, .05], [.9, .95]])
markers = 'v<>^'

a = .5*np.ones((2,))
proj = np.dot(x, a)


def trafo(x, y):
    return sigma(x, 2, -2), sigma(y, 5, 0)


proj_line = np.arange(-50, 50, .02)
proj_transformed_x, proj_transformed_y = trafo(proj_line, proj_line)
proj_x, proj_y = trafo(proj, proj)


a = (x[0] + x[3])/2
b = (x[1] + x[2])/2
c = (a + b)/2
m = (proj_y[3] - proj_y[0])/(proj_x[3] - proj_x[0])

X = np.mean(proj_x) + proj_line
Y = np.mean(proj_y) + m*proj_line


plt.figure()
plt.hold(True)

ms = 10

for i in range(len(x)):
    plt.plot(x[i, 0], x[i, 1], 'g'+markers[i], MarkerSize=ms)
    plt.plot(proj[i], proj[i], 'b'+markers[i], MarkerSize=ms)
    plt.plot(proj_x[i], proj_y[i], 'r'+markers[i], MarkerSize=ms)

dots = 3

plt.plot(proj_line, proj_line, 'k.', MarkerSize=dots)
plt.plot(proj_transformed_x, proj_transformed_y, 'r.', MarkerSize=dots)
plt.plot(X, Y, 'k')

for x in proj_line[::4]:
    a, b = trafo(proj_line, x*np.ones_like(proj_line))
    plt.plot(a, b, 'k')
    a, b = trafo(x*np.ones_like(proj_line), proj_line)
    plt.plot(a, b, 'k')
    #plot(proj_line, y*np.ones_like(proj_line), 'k')


plt.xlim([-.05, 1.05])
plt.ylim([-.05, 1.05])


plt.show()
