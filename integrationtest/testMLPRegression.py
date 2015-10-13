
import warnings
import tempfile
import shutil

from deeplearning.workflow import Workflow
from deeplearning.split import OpTrainTestSplit
from deeplearning.classifiers import OpMLPTrain
from deeplearning.classifiers import OpMLPPredict
from deeplearning.data import OpPickleCache
from deeplearning.data import OpHDF5Cache
from deeplearning.data import OpStreamingHdf5Reader
from deeplearning.features import OpRecent
from deeplearning.report import OpRegressionReport
from deeplearning.targets import OpExponentiallySegmentedPattern

from pylearn2.models import mlp

from deeplearning.data.integrationdatasets import OpNoisySine
from deeplearning.data.integrationdatasets import OpRandomUnitSquare
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

    def testRun(self):
        c = config.copy()
        c["train"] = {"class": OpMLPTrain,
                      "layer_classes": (mlp.Sigmoid, mlp.Sigmoid),
                      "layer_sizes": (20, 10)}
        w = Workflow.build(c, workingdir=self.wd)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.run()
        return w

    def testNormTarget(self):
        c = config.copy()
        c["source"] = {"class": OpRandomUnitSquare,
                       "shape": (10000, 2)}
        c["train"] = {"class": OpMLPTrain,
                      "layer_classes": (mlp.Sigmoid,),
                      "layer_sizes": (1,)}
        c["features"] = {"class": OpFeatures}
        c["target"] = {"class": OpNormTarget}
        w = Workflow.build(c, workingdir=self.wd)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.run()


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--h5file",
                        help="1D hdf5 file formatted as file.h5:path_in_file",
                        default=None)
    parser.add_argument("--diff", action="store_true",
                        help="use np.diff of input",
                        default=False)

    args = parser.parse_args()

    if args.h5file is not None:
        assert ":" in args.h5file, "See help for argument --h5file"
        filename, internal = args.h5file.split(":")
        config["source"] = {"class": OpStreamingHdf5Reader,
                            "filename": filename,
                            "internal_path": internal}

    if args.diff:
        from deeplearning.features import OpDiff
        config["preprocessing"] = [{"class": OpDiff}]

    test = TestMLPRegression()
    test.remove_tempdir = False
    test.setUp()
    try:
        w = test.testRun()
    finally:
        test.tearDown()

    pred = w._predictionCache.Output[...].wait().squeeze()
    target = w._target.Output[...].wait().squeeze()
    orig = w._source.Output[...].wait().squeeze()

    plt.plot(target, 'b')
    plt.plot(pred, 'r+')
    plt.plot(orig, 'k.')
    plt.legend(("ground truth", "prediction", "original data"))
    plt.show()
