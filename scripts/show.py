# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:19:53 2015

@author: burger
"""

import os
import glob
import cPickle as pkl
import warnings

import numpy as np

from deeplearning.tools.serialization import loads
from deeplearning.workflow import Workflow

try:
    from matplotlib import pyplot as plt
except ImportError:
    warnings.warn("can't plot without matplotlib")


def plot_regression(workflow, dir_):
    classifier_file = os.path.join(dir_,
                                   "classifierCache", "OpPickleCache.pkl")
    with open(classifier_file, 'r') as f:
        classifier = pkl.load(f)

    workflow.set_classifier(classifier)

    target_shape = workflow.Target.meta.shape
    assert target_shape[0] == np.prod(target_shape),\
        "expected 1d for regression"
    target = workflow.Target[:].wait().squeeze()
    prediction = workflow.Prediction[:].wait().squeeze()

    plt.figure()
    plt.plot(target, label="ground truth")
    plt.plot(prediction, label="prediction")
    plt.xlabel("time")
    plt.ylabel("target")
    plt.legend()


def plot_convergence(dir_):
    progress_glob = os.path.join(dir_, "train", "progress_*.pkl")
    files = sorted(glob.glob(progress_glob))
    if len(files) == 0:
        warnings.warn("no progress file found")
        return

    def handle_progress_file(progress_file, name):
        with open(progress_file, "r") as f:
            progress = np.asarray(pkl.load(f))
            plt.semilogy(progress, label=name)

    plt.figure()
    plt.hold(True)
    for filename in files:
        name = os.path.basename(filename).split(".")[0]
        handle_progress_file(filename, name)

    plt.xlabel("epochs")
    plt.ylabel("objective")
    plt.legend()


def main(dir_, show_regression=True, show_convergence=True):
    config_file = os.path.join(dir_, "config.json")
    with open(config_file, 'r') as f:
        config = loads(f.read())

    workflow = Workflow.build(config)

    if show_regression:
        plot_regression(workflow, dir_)

    if show_convergence:
        plot_convergence(dir_)

    plt.show()

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("experiment_dir",
                        help="directory containing the experiment")
    parser.add_argument("-r", "--no_regression", action="store_true",
                        default=False,
                        help="don't show regression results")
    parser.add_argument("-p", "--no_progress", action="store_true",
                        default=False,
                        help="don't show progress report")

    args = parser.parse_args()

    main(args.experiment_dir, show_regression=not args.no_regression,
         show_convergence=not args.no_progress)
