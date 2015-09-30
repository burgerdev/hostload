
import os
import logging

from deeplearning.tools import listifyDict
from deeplearning.tools import expandDict
from deeplearning.tools import IncompatibleTargets

from deeplearning.workflow import Workflow


logger = logging.getLogger(__name__)


def runBatch(config, workingdir):
    extended_config = listifyDict(config)
    configs_to_run = list(expandDict(extended_config))

    n = len(configs_to_run)
    digits = 1

    while n >= 10:
        n /= 10.0
        digits += 1

    dir_template = "{:0%dd}" % digits

    for index, config in enumerate(configs_to_run):
        dirname = dir_template.format(index)
        full_path = os.path.join(workingdir, dirname)
        config["workingdir"] = full_path
        try:
            w = Workflow.build(config)
        except IncompatibleTargets:
            logger.info("IncompatibleTargets in run {}".format(dirname))

        try:
            w.run()
        except Exception as err:
            logger.error("Error in run {}:\n\t{}".format(dirname, str(err)))
