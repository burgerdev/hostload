
import tempfile

from tsdl.batch import run_batch

from tsdl.classifiers import OpMLPTrain
from tsdl.classifiers import OpMLPPredict
from tsdl.classifiers.mlp_init import LeastSquaresWeightInitializer
from tsdl.classifiers.mlp_init import PCAWeightInitializer

from tsdl.tools import OpStreamingHdf5Reader

from tsdl.features import OpRecent
from tsdl.features import OpSimpleCombiner
from tsdl.features import OpChain
from tsdl.features import OpExponentialFilter

from tsdl.report import OpRegressionReport

from tsdl.targets import OpExponentiallySegmentedPattern

from tsdl.tools.generic import OpChangeDtype
from tsdl.tools.extensions import ProgressMonitor
from tsdl.tools.fragile_extensions import SignalExtension

from pylearn2.models import mlp


window_size = 64

feature0 = {"class": OpExponentialFilter,
            "window_size": 128}

feature1 = {"class": OpSimpleCombiner,
            "operators": ({"class": OpRecent, "window_size": window_size},
                          )}

features = {"class": OpChain, "operators": (feature0, feature1)}

initializer_choices = ({"class": PCAWeightInitializer},
                       {"class": PCAWeightInitializer},
                       {"class": PCAWeightInitializer},
                       {"class": LeastSquaresWeightInitializer})

extensions = ({"class": ProgressMonitor, "channel": "train_objective"},
              {"class": ProgressMonitor, "channel": "valid_objective"},
              {"class": SignalExtension})

config = {"features": features,
          "target": {"class": OpExponentiallySegmentedPattern,
                     "baseline_size": 20,
                     "num_segments": 1},
          "train": {"class": OpMLPTrain,
                    "layer_classes": (mlp.Sigmoid, mlp.Sigmoid, mlp.Sigmoid,),
                    "layer_sizes": (36, 24, 12),
                    "max_epochs": 20000,  # 10k -> 3:11:39
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
        args.workingdir = tempfile.mkdtemp(prefix="tsdl_batch_")
        print("created temporary working dir: {}".format(args.workingdir))

    main(args.workingdir)
