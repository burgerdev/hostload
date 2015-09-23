"""
This module provides mixins for determining the problem type (classification vs.
regression) and the appropriate error to raise on incompatible types.
"""


class Classification(object):
    pass


class Regression(object):
    pass


class IncompatibleTargets(Exception):
    pass


