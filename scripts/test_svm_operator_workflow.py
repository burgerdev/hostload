
import numpy as np
import vigra
import h5py

from matplotlib import pyplot as plt

import cPickle as pickle

from lazyflow.graph import Graph
from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.operators import OpReorderAxes
from lazyflow.operators.ioOperators import OpStreamingHdf5Reader
from lazyflow.operators.generic import OpSingleChannelSelector
from lazyflow.operators.generic import OpMultiArrayStacker
from lazyflow.operators.valueProviders import OpValueCache

from deeplearning.targets import OpExponentiallySegmentedPattern
from deeplearning.tools.generic import OpNormalize
from deeplearning.targets import OpDiscretize
from deeplearning.targets import OpClassFromOneHot

from deeplearning.features import OpMean
from deeplearning.features import OpLinearWeightedMean
from deeplearning.features import OpFairness
from deeplearning.features import OpRawWindowed

from deeplearning.split import OpTrainTestSplit

from deeplearning.classifiers import OpSVMTrain
from deeplearning.classifiers import OpSVMPredict


class OpTarget(Operator):
    Input = InputSlot()
    Output = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpTarget, self).__init__(*args, **kwargs)
        self.nrm = OpNormalize(parent=self)
        self.esp = OpExponentiallySegmentedPattern(parent=self)
        self.scs = OpSingleChannelSelector(parent=self)
        self.dis = OpDiscretize(parent=self)
        self.coh = OpClassFromOneHot(parent=self)

        slot = self.Input
        for op in [self.nrm, self.esp, self.scs, self.dis, self.coh]:
            op.Input.connect(slot)
            slot = op.Output
        self.Output.connect(self.coh.Output)

        self.nrm.StdDev.setValue(6000)
        self.esp.NumSegments.setValue(4)
        self.esp.BaselineSize.setValue(60)
        self.scs.Index.setValue(3)
        self.dis.NumLevels.setValue(50)

    def setupOutputs(self):
        pass

    def propagateDirty(self, slot, subindex, roi):
        pass

    def execute(self, slot, subindex, roi, result):
        raise NotImplementedError()


class OpFeatures(Operator):
    Input = InputSlot()
    WindowSize = InputSlot(value=360)
    Output = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpFeatures, self).__init__(*args, **kwargs)

        feats = [OpRawWindowed, OpMean, OpLinearWeightedMean, OpFairness]

        self.stacker = OpMultiArrayStacker(parent=self)
        self.stacker.AxisFlag.setValue('c')
        self.stacker.Images.resize(len(feats))
        self.Output.connect(self.stacker.Output)

        for i, feat in enumerate(feats):
            op = feat(parent=self)
            op.WindowSize.connect(self.WindowSize)
            op.Input.connect(self.Input)
            self.stacker.Images[i].connect(op.Output)

    def setupOutputs(self):
        pass

    def propagateDirty(self, slot, subindex, roi):
        pass

    def execute(self, slot, subindex, roi, result):
        raise NotImplementedError()


def buildWorkflow(f, ip, args):
    g = Graph()

    source = OpStreamingHdf5Reader(graph=g)
    source.Hdf5File.setValue(f)
    source.InternalPath.setValue(ip)
    source.OutputImage.meta.axistags = vigra.defaultAxistags('tc')

    datatc = OpSingleChannelSelector(graph=g)
    datatc.Input.connect(source.OutputImage)
    datatc.Index.setValue(1)

    data = OpReorderAxes(graph=g)
    data.Input.connect(datatc.Output)
    data.AxisOrder.setValue('t')

    time = OpSingleChannelSelector(graph=g)
    time.Input.connect(source.OutputImage)
    time.Index.setValue(0)

    target = OpTarget(graph=g)
    target.Input.connect(data.Output)

    features = OpFeatures(graph=g)
    features.Input.connect(data.Output)

    # print("Data: {}".format(data.Output.meta.getTaggedShape()))
    # print("Features: {}".format(features.Output.meta.getTaggedShape()))

    out = features.Output[...].wait()
    out2 = target.Output[...].wait()
    out = np.concatenate((out, out2[:, np.newaxis]), axis=1)
    t = time.Output[...].wait().squeeze()
    t /= 1e6*60*60*24
    leg = ["raw", "mean", "weighted mean", "fairness", "target"]
    plt.plot(t, out)
    plt.legend(leg)

    splitTarget = OpTrainTestSplit(graph=g)
    splitTarget.Input.connect(target.Output)

    splitFeatures = OpTrainTestSplit(graph=g)
    splitFeatures.Input.connect(features.Output)

    train = OpSVMTrain(graph=g)
    train.Train.resize(2)
    train.Train[0].connect(splitFeatures.Train)
    train.Train[1].connect(splitTarget.Train)
    train.Valid.resize(2)
    train.Valid[0].connect(splitFeatures.Valid)
    train.Valid[1].connect(splitTarget.Valid)

    cache = OpValueCache(graph=g)

    if args.resume:
        with open(args.save_path, 'r') as f:
            svc = pickle.load(f)
        x = np.zeros((1,), dtype=np.object)
        x[0] = svc
        cache.Input.setValue(x)
    else:
        cache.Input.connect(train.Classifier)
        with open(args.save_path, 'w') as f:
            svc = cache.Output.value
            pickle.dump(svc, f)
        cache.fixAtCurrent.setValue(True)

    predict = OpSVMPredict(graph=g)
    predict.Classifier.connect(cache.Output)
    predict.Input.connect(splitFeatures.Test)

    prediction = predict.Output[...].wait()[:, np.newaxis] + .1
    ground_truth = splitTarget.Test[...].wait()[:, np.newaxis]
    desc = splitTarget.Description[...].wait()
    to_plot = np.concatenate((prediction, ground_truth), axis=1)
    to_plot_t = t[desc == 2]
    print(to_plot_t.shape)
    print(to_plot.shape)

    plt.figure()
    plt.plot(to_plot_t, to_plot)
    leg = ["predicted", "ground truth"]
    plt.legend(leg)

    plt.show()


def run(args):
    with h5py.File(args.h5file) as h5f:
        buildWorkflow(h5f, args.internal_path, args)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("h5file", help="cpu usage hdf5 file")
    parser.add_argument("internal_path",
                        help="cpu usage hdf5 file internal path")
    parser.add_argument("-s", "--save_path", action="store", type=str,
                        default='/tmp/svm.pkl',
                        help="path for saving svm (default: /tmp/svm.pkl)")
    parser.add_argument("-r", "--resume", action="store_true",
                        default=False,
                        help="start from pickled SVM")

    args = parser.parse_args()
    run(args)
