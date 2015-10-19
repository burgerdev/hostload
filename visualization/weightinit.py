
import numpy as np
from matplotlib import pyplot as plt

from lazyflow.graph import Graph

from pylearn2.models.mlp import MLP
from pylearn2.models.mlp import Sigmoid

from deeplearning.features import OpRecent

from deeplearning.classifiers.mlp import PCAWeightInitializer
from deeplearning.classifiers.mlp import NormalWeightInitializer
from deeplearning.classifiers.mlp import LeastSquaresWeightInitializer

from deeplearning.data.integrationdatasets import OpNoisySine


def plot_line(a, b, *args, **kwargs):
    x = np.linspace(-.5, 1.5)
    if np.abs(a[1]) < 1e-6:
        y = np.zeros_like(x)
    else:
        y = -(b + a[0]*x)/a[1]

    plt.plot(x, y, *args, **kwargs)


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

    plt.figure()
    plt.hold(True)

    xy = features.Output[...].wait()

#    increasing = xy[:, 0] >= xy[:, 1]
#    plt.plot(xy[increasing, 0], xy[increasing, 1], 'rx')
#    plt.plot(xy[~increasing, 0], xy[~increasing, 1], 'bx')
    plt.plot(xy[:, 0], xy[:, 1], 'kx')

    w, b = get_init(features, PCAWeightInitializer, nhid=10)

    for i in range(len(b)):
        plot_line(w[:, i], b[i], label=str(i))

    plt.axis([-.5, 1.5, -.5, 1.5])
    plt.axis('equal')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
