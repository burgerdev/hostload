# -*- coding: utf-8 -*-

import warnings
import tempfile

from datetime import datetime
from collections import OrderedDict


try:
    from matplotlib import pyplot as plt
except ImportError:
    _plot_available = False
else:
    _plot_available = True

from pylearn2.models import mlp


from tsdl.data.integrationdatasets import OpMackeyGlass


# workflow
from tsdl.workflow import Workflow

# features
from tsdl.features import OpRecent
from tsdl.targets import OpExponentiallySegmentedPattern
from tsdl.split import OpTrainTestSplit
from tsdl.report import OpRegressionReport

# classifiers
from tsdl.classifiers import OpSVMTrain
from tsdl.classifiers import OpSVMPredict
from tsdl.classifiers import OpMLPTrain
from tsdl.classifiers import OpMLPPredict
from tsdl.classifiers.mlp_init import LeastSquaresWeightInitializer
from tsdl.classifiers.mlp_init import PCAWeightInitializer
from tsdl.classifiers.mlp_init import StandardWeightInitializer

# caches
from tsdl.data import OpPickleCache
from tsdl.data import OpHDF5Cache

# train extensions
from tsdl.tools.extensions import ProgressMonitor


# options available for initialization
_train_choice = OrderedDict()
_train_choice["random"] = {"class": StandardWeightInitializer}
_train_choice["pca"] = tuple([{"class": init}
                              for init in (PCAWeightInitializer,
                                           PCAWeightInitializer,
                                           LeastSquaresWeightInitializer)])
_train_choice["grid"] = None
_train_choice["svm"] = None


def _get_conf(args):
    if args.mode == "svm":
        train = {"class": OpSVMTrain}
        predict = {"class": OpSVMPredict}

    else:
        if args.mode not in _train_choice or _train_choice[args.mode] is None:
            raise NotImplementedError(
                "mode '{}' not implemented yet".format(args.mode))
        train = {"class": OpMLPTrain,
                 "weight_initializer": _train_choice[args.mode],
                 "layer_sizes": (100, 10),
                 "layer_classes": (mlp.Sigmoid,),
                 "max_epochs": args.epochs,
                 "terminate_early": False,
                 "extensions": ({"class": ProgressMonitor,
                                 "channel": "train_objective"},)}
        predict = {"class": OpMLPPredict}

    config = {"class": Workflow,
              "source": {"class": OpMackeyGlass,
                         "shape": (10000,)},
              "features": {"class": OpRecent, "window_size": 8},
              "target": {"class": OpExponentiallySegmentedPattern,
                         "baseline_size": 8,
                         "num_segments": 1},
              "split": {"class": OpTrainTestSplit},
              "classifierCache": {"class": OpPickleCache},
              "train": train,
              "predict": predict,
              "predictionCache": {"class": OpHDF5Cache},
              "report": {"class": OpRegressionReport,
                         "levels": 50}}

    return config


def main(args):
    conf = _get_conf(args)

    tempdir = args.workingdir
    if tempdir is None:
        prefix = "{}_{:%Y-%m-%d_%H-%M}".format(args.mode, datetime.now())
        tempdir = tempfile.mkdtemp(prefix=prefix)

    workflow = Workflow.build(conf, workingdir=tempdir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        workflow.run()

    if args.plot:
        assert _plot_available, "matplotlib is needed for option --plot"
        ground_truth = workflow.Target[...].wait()
        prediction = workflow.Prediction[...].wait()
        plt.figure()
        plt.hold(True)
        plt.plot(ground_truth, 'b')
        plt.plot(prediction, 'r')
        plt.hold(False)
        plt.show()
    print("output written to dir {}".format(tempdir))


if __name__ == "__main__":
    from argparse import ArgumentParser
    help_text = """
This script can be used to demonstrate the differences in training a neural
network with either random initialization or a more sophisticated
initialization. For comparison to non-NN approaches, SVM regression is
available.
"""
    default_epochs = 100
    mode_help = "use this learning tool (options: {})".format(
        ", ".join(_train_choice.keys()))

    parser = ArgumentParser(description=help_text)
    parser.add_argument("-d", "--workingdir", action="store", default=None,
                        help="working directory (default: temp dir)")
    parser.add_argument("-m", "--mode", action="store", default="svm",
                        help=mode_help)
    parser.add_argument("-p", "--plot", action="store_true",
                        help="plot results (default: don't plot)",
                        default=False)
    parser.add_argument("-e", "--epochs", action="store", type=int,
                        help="max epochs (default: {})".format(default_epochs),
                        default=default_epochs, metavar="<int>")

    args = parser.parse_args()
    main(args)
