
import tempfile

from matplotlib import pyplot as plt
import matplotlib.animation as animation

import numpy as np
import theano

from pylearn2.models import mlp

from deeplearning.data.integrationdatasets import OpXORTarget
from deeplearning.data.integrationdatasets import OpRandomCorners
from deeplearning.data.integrationdatasets import OpFeatures

from deeplearning.tools.extensions import WeightKeeper
from deeplearning.tools.extensions import ProgressMonitor
from deeplearning.tools.extensions import BuildableTrainExtension

from deeplearning.workflow import RegressionWorkflow
from deeplearning.split import OpTrainTestSplit
from deeplearning.classifiers import OpMLPTrain
from deeplearning.classifiers import OpMLPPredict
from deeplearning.classifiers.mlp_init import LeastSquaresWeightInitializer
from deeplearning.classifiers.mlp_init import StandardWeightInitializer
from deeplearning.classifiers.mlp_init import PCAWeightInitializer


num_epochs = 200
num_plots = 6

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

fig1 = plt.figure()

data = np.random.rand(2, 25)
l, = plt.plot([], [], 'r-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x')
plt.title('test')
line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
                                   interval=50, blit=True)
#line_ani.save('lines.mp4')

fig2 = plt.figure()

x = np.arange(-9, 10)
y = np.arange(-9, 10).reshape(-1, 1)
base = np.hypot(x, y)
ims = []
for add in np.arange(15):
    ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))

im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
                                   blit=True)
#im_ani.save('im.mp4', metadata={'artist':'Guido'})

plt.show()
"""


class PlotExtension(BuildableTrainExtension):
    def on_monitor(self, model, dataset, algorithm):
        """
        save the model's weights
        """
        A, b = model.layers[0].get_param_values()
        self.adjust_plot(A, b)

    def setup(self, model, dataset, algorithm):
        """
        initialize the weight list
        """
        self._fig = plt.figure()
        self._stills = []
        t = np.linspace(0, 1, 25)
        X, Y = np.meshgrid(t, t)
        x = X.reshape((-1, 1))
        y = Y.reshape((-1, 1))
        v = np.concatenate((x, y), axis=1).T
        self._v = v

    def adjust_plot(self, A, b):
        w = np.dot(A, self._v) + b[:, np.newaxis]
        w = 1./(1 + np.exp(-w))
        p = plt.plot(w[0, :], w[1, :], 'k.')

        self._stills.append(p)

    def get_animation(self):
        return animation.ArtistAnimation(self._fig, self._stills, interval=100,
                                         repeat_delay=500, blit=True)
        

config = {"source": {"class": OpRandomCorners,
                     "shape": (10000, 2)},
          "features": {"class": OpFeatures},
          "target": {"class": OpXORTarget,},
          "split": {"class": OpTrainTestSplit},
          "train": {"class": OpMLPTrain,
                    "max_epochs": num_epochs,
                    #"weight_initializer": ({"class": NormalWeightInitializer},
                    #"weight_initializer": ({"class": PCAWeightInitializer},
                    #                       {"class": LeastSquaresWeightInitializer}),
                    "weight_initializer": ({"class": PCAWeightInitializer},
                                           {"class": StandardWeightInitializer}),
                    "learning_rate": .2,
                    "layer_sizes": (2,),
                    "layer_classes": (mlp.Sigmoid,)},
          "predict": {"class": OpMLPPredict}}


def main():

    ext1 = {"class": WeightKeeper}
    ext2 = {"class": ProgressMonitor}
    ext3 = {"class": PlotExtension}
    config["train"]["extensions"] = (ext1, ext2, ext3)
    wd = tempfile.mkdtemp(prefix="xor_")
    w = RegressionWorkflow.build(config, workingdir=wd)
    w.run()

    plt_ext = filter(lambda x: isinstance(x, PlotExtension),
                     w._train.extensions_used)[0]
    ani = plt_ext.get_animation()
    # ani.save('im.mp4', metadata={'artist':'Guido'})

    plt.show()


if __name__ == "__main__":
    main()
