
import tempfile

from deeplearning.batch import run_batch

from deeplearning.classifiers import OpMLPTrain
from deeplearning.classifiers import OpMLPPredict
from deeplearning.classifiers.mlp import LeastSquaresWeightInitializer
from deeplearning.classifiers.mlp import PCAWeightInitializer

from deeplearning.data import OpStreamingHdf5Reader

from deeplearning.features import OpRecent
from deeplearning.features import OpSimpleCombiner
from deeplearning.features import OpChain
from deeplearning.features import OpExponentialFilter

from deeplearning.report import OpRegressionReport

from deeplearning.targets import OpExponentiallySegmentedPattern

from deeplearning.tools.generic import OpChangeDtype
from deeplearning.tools.extensions import ProgressMonitor
from deeplearning.tools.fragile_extensions import SignalExtension

from pylearn2.models import mlp


window_size = 32

feature0 = {"class": OpExponentialFilter,
            "window_size": 32}

feature1 = {"class": OpSimpleCombiner,
            "operators": ({"class": OpRecent, "window_size": window_size},
                          )}

features = {"class": OpChain, "operators": (feature0, feature1)}

initializer_choices = ({"class": PCAWeightInitializer},
                       {"class": PCAWeightInitializer},
                       {"class": LeastSquaresWeightInitializer})

extensions = ({"class": ProgressMonitor, "channel": "train_objective"},
              {"class": ProgressMonitor, "channel": "valid_objective"},
              {"class": SignalExtension})

config = {"features": features,
          "target": {"class": OpExponentiallySegmentedPattern,
                     "baseline_size": 8,
                     "num_segments": 1},
          "train": {"class": OpMLPTrain,
                    "layer_classes": (mlp.Sigmoid, mlp.Sigmoid,),
                    "layer_sizes": (24, 16),
                    "max_epochs": 10000,  # 10k -> 3:11:39
                    "terminate_early": False,
                    "learning_rate": 0.35,
                    "weight_initializer": initializer_choices,
                    "continue_learning": True,
                    "extensions": extensions},
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
