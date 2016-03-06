"""
This module provides mixins for determining the problem type (classification
vs. regression) and the appropriate error to raise on incompatible types.
Mixing classification and regression in the same workflow is illegal and
results in an IncompatibleTargets exception being thrown.
"""


# these are identifier mixins, no need for public methods
# pylint: disable=R0903

class Classification(object):
    """
    classify as one of N target classes
    """
    pass


class Regression(object):
    """
    approximate a real-valued function
    """
    pass


class IncompatibleTargets(Exception):
    """
    thrown whenever classification and regression are mixed
    """
    pass
