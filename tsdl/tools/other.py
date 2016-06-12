"""
This really did not fit in elsewhere
"""

import numpy as np


def get_rng():
    """
    get a RNG that is arbitrary, but repeatable

    Use this for testing or publishing.
    """
    return np.random.RandomState(420)
