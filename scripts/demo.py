
import sys

import numpy as np
import vigra

from lazyflow.graph import Graph
from lazyflow.utility.testing import OpArrayPiperWithAccessCount
from lazyflow.operators import OpReorderAxes

from deeplearning.workflow import Workflow
from deeplearning.split import OpTrainTestSplit
from deeplearning.classifiers import OpStateTrain
from deeplearning.classifiers import OpStatePredict
from deeplearning.classifiers import OpSVMTrain
from deeplearning.classifiers import OpSVMPredict
from deeplearning.classifiers import OpRFTrain
from deeplearning.classifiers import OpRFPredict
from deeplearning.classifiers import OpMLPTrain
from deeplearning.classifiers import OpMLPPredict
from deeplearning.classifiers import OpDeepTrain
from deeplearning.data import OpPickleCache
from deeplearning.data import OpHDF5Cache
from deeplearning.report import OpClassificationReport

from pylearn2.models.mlp import Sigmoid
from pylearn2.models.mlp import RectifiedLinear


class OpSource(OpArrayPiperWithAccessCount):
    @staticmethod
    def build(d, graph=None, parent=None, workingdir=None):
        assert "shape" in d
        shape = d["shape"]
        np.random.seed(420)
        data = np.random.randint(0, 2, size=shape).astype(np.float32)
        rnd = np.random.random(size=shape)*.05
        rnd = np.where(data > .5, -rnd, rnd).astype(np.float32)
        data += rnd
        assert data.min() >= 0 and data.max() <= 1
        tags = "tc"
        data = vigra.taggedView(data, axistags=tags)
        op = OpSource(parent=parent, graph=graph)
        op.Input.setValue(data)
        return op


class OpFeatures(OpReorderAxes):
    @staticmethod
    def build(d, graph=None, parent=None, workingdir=None):
        op = OpFeatures(parent=parent, graph=graph)
        op.AxisOrder.setValue('tc')
        return op


class OpTarget(OpArrayPiperWithAccessCount):
    @staticmethod
    def build(d, graph=None, parent=None, workingdir=None):
        op = OpTarget(parent=parent, graph=graph)
        return op

    def setupOutputs(self):
        assert len(self.Input.meta.shape) == 2
        self.Output.meta.shape = (self.Input.meta.shape[0], 2)
        self.Output.meta.dtype = np.float
        self.Output.meta.axistags = vigra.defaultAxistags('tc')

    def execute(self, slot, subindex, roi, result):
        data = self.Input[roi.start[0]:roi.stop[0], :].wait()
        data = np.round(data).astype(np.bool)
        for i, c in enumerate(range(roi.start[1], roi.stop[1])):
            result[:, i] = np.where(data[:, 0] ^ data[:, 1], c, 1-c)


config = {"class": Workflow,
          "source": {"class": OpSource,
                     "shape": (1000, 2)},
          "features": {"class": OpFeatures},
          "target": {"class": OpTarget},
          "split": {"class": OpTrainTestSplit},
          "classifierCache": {"class": OpPickleCache},
          "predict": {"class": OpMLPPredict},
          "predictionCache": {"class": OpHDF5Cache},
          "report": {"class": OpClassificationReport}}


def getConfig(args):
    c = config.copy()
    if args.mlp:
        c["train"] = {"class": OpMLPTrain,
                      #"layer_classes": (RectifiedLinear,),
                      "layer_classes": (Sigmoid,),
                      "layer_sizes": 2}
    elif args.dnn:
        c["train"] = {"class": OpDeepTrain,
                      "num_hidden_layers": 1,
                      "size_hidden_layers": 4}
    elif args.svm:
        raise NotImplementedError("not implemented yet")
    elif args.rf:
        c["train"] = {"class": OpRFTrain}
        c["predict"] = {"class": OpRFPredict}
    else:
        print("No classifier provided")
        sys.exit(1)
    return c

def run(args):
    c = getConfig(args)
    c["workingdir"] = args.workingdir
    w = Workflow.build(c)
    w.run()
    print("Working directory: {}".format(w._workingdir))


def show():
    from matplotlib import pyplot as plt
    g = Graph()
    source = OpSource.build(config["source"], graph=g)
    target = OpTarget(graph=g)
    target.Input.connect(source.Output)

    X = source.Output[...].wait()
    y = target.Output[...].wait()
    y = np.argmax(y, axis=1).astype(np.bool)

    red = X[y]
    blue = X[~y]

    plt.hold(True)
    plt.plot(red[:, 0], red[:, 1], 'r+')
    plt.plot(blue[:, 0], blue[:, 1], 'b+')
    plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-d", "--workingdir", help="storage directory",
                        default=None)
    parser.add_argument("-s", "--show", action="store_true",
                        default=False, help="show dataset")
    parser.add_argument("--mlp", action="store_true",
                        default=False, help="use MLP classifier")
    parser.add_argument("--dnn", action="store_true",
                        default=False,
                        help="use deep learning classifier")
    parser.add_argument("--svm", action="store_true",
                        default=False, help="use SVM classifier")
    parser.add_argument("--rf", action="store_true",
                        default=False,
                        help="use RandomForest classifier")


    args = parser.parse_args()

    if args.show:
        show()
    else:
        run(args)
