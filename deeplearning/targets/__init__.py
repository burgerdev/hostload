"""
Operators that construct a target from given features.
"""

from .segmented import OpExponentiallySegmentedPattern
from .converters import OpDiscretize
from .converters import OpClassFromOneHot
from .hostload import OpHostloadTarget
