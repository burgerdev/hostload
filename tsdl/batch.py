"""
tools for batch processing

Batch runs are specified by the special dictionary structure detailed in module
'tools/config.py'.
"""

import os
import logging
import traceback

from tsdl.tools import listify_dict
from tsdl.tools import expand_dict
from tsdl.tools import IncompatibleTargets

from tsdl.workflow import Workflow


LOGGER = logging.getLogger(__name__)


# we want to catch all exceptions, don't complain
# pylint: disable=W0703
def run_batch(config, workingdir, continue_on_failure=True):
    """
    run a batch configuration inspecified directory

    * config is a tsdl.workflow.Workflow configuration with lists where
      multiple parameters should be run
      e.g.: config = {'a': 0, 'b': [1, 2]} results in 'b' mapping to 1 in the
            first run, to 2 in the second run
    * the directory has to exist
    """
    configs_to_run = list(expand_dict(listify_dict(config)))

    num_configs = len(configs_to_run)
    digits = 1

    while num_configs >= 10:
        num_configs /= 10.0
        digits += 1

    dir_template = "{:0%dd}" % digits

    for index, config in enumerate(configs_to_run):
        dirname = dir_template.format(index)
        config["workingdir"] = os.path.join(workingdir, dirname)
        try:
            workflow = Workflow.build(config)
            workflow.run()
        except IncompatibleTargets:
            LOGGER.info("IncompatibleTargets in run %s", dirname)
            err_file_name = os.path.join(config["workingdir"], "SKIPPED")
            err_msg = "IncompatibleTargets raised"
        except Exception as err:
            LOGGER.error("Error in run %s:\n\t%s", dirname, str(err))
            err_file_name = os.path.join(config["workingdir"], "FAILURE")
            err_msg = traceback.format_exc()
            if not continue_on_failure:
                raise
        else:
            err_file_name = os.path.join(config["workingdir"], "SUCCESS")
            err_msg = ""

        with open(err_file_name, "w") as err_file:
            err_file.write(err_msg)
