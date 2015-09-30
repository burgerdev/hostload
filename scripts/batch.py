
import tempfile

from deeplearning.batch import runBatch

from deeplearning.classifiers import OpMLPTrain
from deeplearning.classifiers import OpMLPPredict

from deeplearning.data import OpStreamingHdf5Reader

from deeplearning.features import OpRecent

from deeplearning.report import OpRegressionReport

from deeplearning.targets import OpExponentiallySegmentedPattern

from pylearn2.models import mlp


config = {"features": {"class": OpRecent,
                       "window_size": [1, 2, 4, 8, 16, 32]},
          "target": {"class": OpExponentiallySegmentedPattern,
                     "baseline_size": [8, 64, 256, 1024],
                     "num_segments": 1},
          "train": {"class": OpMLPTrain,
                    "layer_classes": (mlp.Sigmoid, mlp.Sigmoid),
                    "layer_sizes": (20, 10)},
          "predict": {"class": OpMLPPredict},
          "report": {"class": OpRegressionReport}}


def main(workingdir):
    runBatch(config, workingdir)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("h5file",
                        help="1D hdf5 file formatted as file.h5:path_in_file")
    parser.add_argument("-d", "--workingdir", default=None,
                        help="workingdir (will be a temporary dir if omitted)")

    args = parser.parse_args()

    assert ":" in args.h5file, "See help for argument <h5file>"
    filename, internal = args.h5file.split(":")
    config["source"] = {"class": OpStreamingHdf5Reader,
                        "filename": filename,
                        "internal_path": internal}

    if args.workingdir is None:
        args.workingdir = tempfile.mkdtemp(prefix="deeplearning_batch_")

    main(args.workingdir)
