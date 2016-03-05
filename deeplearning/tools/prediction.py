"""
This module provides mixins for determining the problem type (classification vs.
regression) and the appropriate error to raise on incompatible types.
"""


# these are identifier mixins, no need for public methods
# pylint: disable=R0903

class Classification(object):
    pass


class Regression(object):
    pass


class IncompatibleTargets(Exception):
    pass


