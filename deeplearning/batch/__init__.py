"""
tools for batch processing
"""

import os
import logging

from deeplearning.tools import listifyDict
from deeplearning.tools import expandDict
from deeplearning.tools import IncompatibleTargets

from deeplearning.workflow import Workflow


LOGGER = logging.getLogger(__name__)


def run_batch(config, workingdir, continue_on_failure=True):
    """
    run a batch configuration inspecified directory

    * config is a deeplearning.workflow.Workflow configuration with lists where
      multiple parameters should be run
      e.g.: config = {'a': 0, 'b': [1, 2]} results in 'b' mapping to 1 in the
            first run, to 2 in the second run
    * the directory has to exist
    """
    extended_config = listifyDict(config)
    configs_to_run = list(expandDict(extended_config))

    num_configs = len(configs_to_run)
    digits = 1

    while num_configs >= 10:
        num_configs /= 10.0
        digits += 1

    dir_template = "{:0%dd}" % digits

    for index, config in enumerate(configs_to_run):
        dirname = dir_template.format(index)
        full_path = os.path.join(workingdir, dirname)
        config["workingdir"] = full_path
        try:
            workflow = Workflow.build(config)
        except IncompatibleTargets:
            LOGGER.info("IncompatibleTargets in run %s", dirname)

        try:
            workflow.run()
        except Exception as err:
            LOGGER.error("Error in run %s:\n\t%s", dirname, str(err))
            if not continue_on_failure:
                raise
