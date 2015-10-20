# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:19:53 2015

@author: burger
"""

import os
import cPickle as pkl

import numpy as np

from deeplearning.tools.serialization import loads
from deeplearning.workflow import Workflow


def main(dir_):
    from matplotlib import pyplot as plt
    config_file = os.path.join(dir_, "config.json")
    with open(config_file, 'r') as f:
        config = loads(f.read())

    workflow = Workflow.build(config)

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

    plt.plot(target, label="ground truth")
    plt.plot(prediction, label="prediction")
    plt.show()

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("experiment_dir",
                        help="directory containing the experiment")

    args = parser.parse_args()

    main(args.experiment_dir)
