
import tempfile

from deeplearning.batch import run_batch

from deeplearning.classifiers import OpMLPTrain
from deeplearning.classifiers import OpMLPPredict
from deeplearning.classifiers.mlp import NormalWeightInitializer
from deeplearning.classifiers.mlp import LeastSquaresWeightInitializer
from deeplearning.classifiers.mlp import PCAWeightInitializer

from deeplearning.data import OpStreamingHdf5Reader

from deeplearning.features import OpRecent
from deeplearning.features import OpSimpleCombiner
from deeplearning.features import OpMean
from deeplearning.features import OpFairness
from deeplearning.features import OpLinearWeightedMean
from deeplearning.features import OpDiff

from deeplearning.report import OpRegressionReport

from deeplearning.targets import OpExponentiallySegmentedPattern

from deeplearning.tools.generic import OpChangeDtype

from pylearn2.models import mlp


window_size = 64

features = {"class": OpSimpleCombiner,
            "operators": ({"class": OpRecent, "window_size": window_size},
#                          {"class": OpMean, "window_size": window_size},
#                          {"class": OpLinearWeightedMean,
#                           "window_size": window_size},
#                          {"class": OpFairness, "window_size": window_size},
#                          {"class": OpDiff})
                          )}


initializer_choices = [(this, {"class": NormalWeightInitializer})
                       for this in (LeastSquaresWeightInitializer,
                                    PCAWeightInitializer)]


config = {"features": features,
          "target": {"class": OpExponentiallySegmentedPattern,
                     "baseline_size": [8, 16, 32, 64],
                     "num_segments": 1},
          "train": {"class": OpMLPTrain,
                    "layer_classes": (mlp.Sigmoid,),
                    "layer_sizes": [8, 16, 32],
                    "max_epochs": 5000,
                    "terminate_early": False,
                    "learning_rate": 0.2,
                    "weight_initializer": initializer_choices},
          "predict": {"class": OpMLPPredict},
          "report": {"class": OpRegressionReport},
          "preprocessing": (OpChangeDtype,)}


def main(workingdir):
    run_batch(config, workingdir, continue_on_failure=False)


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
        print("created temporary working dir: {}".format(args.workingdir))

    main(args.workingdir)
