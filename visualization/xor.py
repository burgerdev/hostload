
import tempfile

import numpy as np
import theano
from pylearn2.models import mlp

from deeplearning.data.integrationdatasets import OpXORTarget
from deeplearning.data.integrationdatasets import OpRandomUnitSquare
from deeplearning.data.wrappers import OpArrayPiper
from deeplearning.data import OpPickleCache
from deeplearning.data import OpHDF5Cache

from deeplearning.tools.extensions import WeightKeeper

from deeplearning.workflow import Workflow
from deeplearning.split import OpTrainTestSplit
from deeplearning.classifiers import OpMLPTrain
from deeplearning.classifiers import OpMLPPredict
from deeplearning.classifiers.mlp import LeastSquaresWeightInitializer
from deeplearning.classifiers.mlp import NormalWeightInitializer
from deeplearning.classifiers.mlp import IdentityWeightInitializer
from deeplearning.classifiers.mlp import PCAWeightInitializer
from deeplearning.report import OpRegressionReport


num_epochs = 10
num_plots = 6


config = {"class": Workflow,
          "source": {"class": OpRandomUnitSquare,
                     "shape": (10000, 2)},
          "features": {"class": OpArrayPiper},
          "target": {"class": OpXORTarget,},
          "split": {"class": OpTrainTestSplit},
          "classifierCache": {"class": OpPickleCache},
          "train": {"class": OpMLPTrain,
                    "max_epochs": num_epochs,
                    #"weight_initializer": ({"class": NormalWeightInitializer},
                    "weight_initializer": ({"class": PCAWeightInitializer},
                                           {"class": LeastSquaresWeightInitializer}),
                    "learning_rate": .2,
                    "layer_sizes": (2,),
                    "layer_classes": (mlp.Sigmoid,)},
          "predict": {"class": OpMLPPredict},
          "predictionCache": {"class": OpHDF5Cache},
          "report": {"class": OpRegressionReport,
                     "levels": 50}}


def hyperplane(X, Y, w):
    a = w[0]
    b = w[1]

    Z = X*a[0] + Y*a[1] + b
    return Z


def main():
    n = 10
    t = np.linspace(0, 1, n)
    X, Y = np.meshgrid(t, t)
    data = np.concatenate((X.reshape(n**2, 1), Y.reshape(n**2, 1)), axis=1)
    data = data.astype(np.float32)
    target = 1 - np.square(1 - data.sum(axis=1))

    ext = {"class": WeightKeeper}
    config["train"]["extensions"] = (ext,)
    wd = tempfile.mkdtemp(prefix="xor_")
    w = Workflow.build(config, workingdir=wd)
    w.run()

    wk = filter(lambda obj: isinstance(obj, WeightKeeper),
                w._train.extensions_used)[0]
    weights = wk.get_weights()

    mlp.Layer
    model = mlp.MLP([mlp.Sigmoid(dim=2, irange=.1, layer_name="a")], nvis=2)

    from matplotlib import pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    plt.hold(True)

    for i, w in enumerate(weights):
        if i < num_plots - 1 or i == num_epochs - 1:
            model.layers[0].set_param_values(w[0:2])
            shared = theano.shared(data)
            y = model.fprop(shared).eval()
            axes.plot3D(y[:, 0], y[:, 1], target, '.', label="{}".format(i+1))
    Z = hyperplane(X, Y, w[2:])
    axes.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0,
                      facecolors=cm.jet(Z), shade=False)
    plt.legend(loc="lower left")
    plt.axis([-.1, 1.1, -.1, 1.1])
    plt.show()

if __name__ == "__main__":
    main()
