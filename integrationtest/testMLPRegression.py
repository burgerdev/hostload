
import warnings
import tempfile
import shutil

from matplotlib import pyplot as plt

from deeplearning.workflow import Workflow
from deeplearning.split import OpTrainTestSplit
from deeplearning.classifiers import OpMLPTrain
from deeplearning.classifiers import OpMLPPredict
from deeplearning.classifiers.mlp import NormalWeightInitializer
from deeplearning.classifiers.mlp import LeastSquaresWeightInitializer
from deeplearning.classifiers.mlp import FilterWeightInitializer
from deeplearning.classifiers.mlp import PCAWeightInitializer
from deeplearning.data import OpPickleCache
from deeplearning.data import OpHDF5Cache
from deeplearning.data import OpStreamingHdf5Reader
from deeplearning.features import OpRecent
from deeplearning.report import OpRegressionReport
from deeplearning.targets import OpExponentiallySegmentedPattern

from pylearn2.models import mlp

from deeplearning.data.integrationdatasets import OpNoisySine
from deeplearning.data.integrationdatasets import OpRandomUnitSquare
from deeplearning.data.integrationdatasets import OpMackeyGlass
from deeplearning.data.integrationdatasets import OpNormTarget
from deeplearning.data.integrationdatasets import OpFeatures


config = {"class": Workflow,
          "source": {"class": OpNoisySine,
                     "shape": (10000,)},
          "features": {"class": OpRecent, "window_size": 30},
          "target": {"class": OpExponentiallySegmentedPattern,
                     "baseline_size": 10,
                     "num_segments": 1},
          "split": {"class": OpTrainTestSplit},
          "classifierCache": {"class": OpPickleCache},
          "train": {"class": OpMLPTrain,
                    "max_epochs": 20},
          "predict": {"class": OpMLPPredict},
          "predictionCache": {"class": OpHDF5Cache},
          "report": {"class": OpRegressionReport,
                     "levels": 50}}


class TestMLPRegression(object):
    remove_tempdir = True
    # remove_tempdir = False
    def setUp(self):
        self.wd = tempfile.mkdtemp(prefix="MLPReg_")

    def tearDown(self):
        if self.remove_tempdir:
            shutil.rmtree(self.wd)
        else:
            import sys
            sys.stderr.write("testMLP: {}\n".format(self.wd))

    def testRun(self, plot=False):
        c = config.copy()
        c["train"]["layer_sizes"] = (20, 10)
        c["train"]["layer_classes"] = (mlp.Sigmoid, mlp.Sigmoid)
        w = Workflow.build(c, workingdir=self.wd)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.run()
        return w

    def testNormTarget(self, plot=False):
        c = config.copy()
        c["source"] = {"class": OpRandomUnitSquare,
                       "shape": (10000, 2)}
        c["train"]["layer_sizes"] = (1,)
        c["train"]["layer_classes"] = (mlp.Sigmoid,)
        c["features"] = {"class": OpFeatures}
        c["target"] = {"class": OpNormTarget}
        w = Workflow.build(c, workingdir=self.wd)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.run()

    def testMG(self, plot=False):
        c = config.copy()
        c["source"] = {"class": OpMackeyGlass}
        c["train"]["layer_sizes"] = (5,)
        c["train"]["layer_classes"] = (mlp.Sigmoid,)
        init_1 = {"class": LeastSquaresWeightInitializer}
        init_2 = {"class": NormalWeightInitializer}
        c["train"]["weight_initializer"] = (init_1, init_2)
        c["train"]["learning_rate"] = .2
        c["features"] = {"class": OpRecent, "window_size": 16}
        c["target"] = {"class": OpExponentiallySegmentedPattern,
                       "baseline_size": 8,
                       "num_segments": 1}
        w = Workflow.build(c, workingdir=self.wd)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.run()

        if plot:
            plt.figure()
            pred = w._predictionCache.Output[...].wait().squeeze()
            target = w._target.Output[...].wait().squeeze()
            orig = w._source.Output[...].wait().squeeze()
            plt.plot(target, 'b')
            plt.plot(pred, 'r+')
            plt.plot(orig, 'k.')
            plt.legend(("ground truth", "prediction", "original data"))



if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--h5file",
                        help="1D hdf5 file formatted as file.h5:path_in_file",
                        default=None)
    parser.add_argument("--norm", action="store_true",
                        help="run test on OpNormTarget",
                        default=False)
    parser.add_argument("--mg", action="store_true",
                        help="run test on Mackey-Glass target",
                        default=False)
    parser.add_argument("-p", "--plot", action="store_true",
                        help="plot results",
                        default=False)
    parser.add_argument("-e", "--epochs", action="store", type=int,
                        help="max epochs", default=100)

    args = parser.parse_args()

    if args.h5file is not None:
        assert ":" in args.h5file, "See help for argument --h5file"
        filename, internal = args.h5file.split(":")
        config["source"] = {"class": OpStreamingHdf5Reader,
                            "filename": filename,
                            "internal_path": internal}

    config["train"]["max_epochs"] = args.epochs

    test = TestMLPRegression()
    test.remove_tempdir = False

    tests = {"norm": test.testNormTarget, "mg": test.testMG}

    for key in tests:
        if not getattr(args, key, False):
            continue
        test.setUp()
        try:
            w = tests[key](plot=args.plot)
        finally:
            test.tearDown()

    if args.plot:
        plt.show()

