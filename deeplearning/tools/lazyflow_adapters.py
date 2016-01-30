"""
convenience wrappers for lazyflow classes
"""

import logging

from lazyflow.operator import Operator as _Operator
from lazyflow.operator import InputSlot, OutputSlot
from lazyflow.rtype import SubRegion
from lazyflow.operators import OpReorderAxes


LOGGER = logging.getLogger(__name__)


class Operator(_Operator):
    def setInSlot(self, slot, subindex, key, value):
        LOGGER.error("'setInSlot' not supported in this module")
