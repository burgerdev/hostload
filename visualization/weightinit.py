
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lazyflow.graph import Graph

from pylearn2.models.mlp import MLP
from pylearn2.models.mlp import Sigmoid

from deeplearning.features import OpRecent

from deeplearning.classifiers.mlp import PCAWeightInitializer
from deeplearning.classifiers.mlp import NormalWeightInitializer
from deeplearning.classifiers.mlp import LeastSquaresWeightInitializer

from deeplearning.data.integrationdatasets import OpNoisySine


def plot_hyperplane(axes, a, b, *args, **kwargs):
    x = np.linspace(-.5, 1.5, 20)
    X, Y = np.meshgrid(x, x)

    def sigmoid(x, y):
        return 1/(1 + np.exp((a[0]*x + a[1]*y + b)))

    axes.plot_wireframe(X, Y, sigmoid(X, Y), *args, **kwargs)


def get_init(features, initializer_class, nhid=5):
    initializer = initializer_class.build({}, graph=features.graph)
    if hasattr(initializer, "Input"):
        initializer.Input.resize(2)
        initializer.Input[0].connect(features.Output)
    nvis = features.Output.meta.shape[1]
    layer = Sigmoid(layer_name="a", irange=0, dim=nhid)
    # layer needs to be initialized by MLP first
    MLP(layers=[layer], nvis=nvis)
    initializer.init_layer(layer, nvis=nvis, nhid=nhid)
    weights = layer.get_weights()
    biases = layer.get_biases()
    return weights, biases


def main():
    g = Graph()
    source = OpNoisySine.build({}, graph=g)
    features = OpRecent.build({"window_size": 2}, graph=g)
    features.Input.connect(source.Output)

    fig = plt.figure()
    axes_1 = fig.add_subplot(121, projection='3d')
    axes_2 = fig.add_subplot(122, projection='3d')
    axes = [axes_1, axes_2]
    plt.hold(True)

    xy = features.Output[...].wait()

#    increasing = xy[:, 0] >= xy[:, 1]
#    plt.plot(xy[increasing, 0], xy[increasing, 1], 'rx')
#    plt.plot(xy[~increasing, 0], xy[~increasing, 1], 'bx')
    for i in range(2):
        axes[i].plot(xy[::10, 0], xy[::10, 1], .5, 'kx')

    w, b = get_init(features, PCAWeightInitializer, nhid=4)
    colors = 'rb'

    for i in range(len(b)):
        ci = i % len(colors)
        ax = axes[1 if i > 1 else 0]
        plot_hyperplane(ax, w[:, i], b[i], color=colors[ci], label=str(i))

    #plt.axis([-.5, 1.5, -.5, 1.5])
    #plt.axis('equal')
    #plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
