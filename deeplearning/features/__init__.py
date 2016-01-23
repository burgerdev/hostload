"""
module for time series features

Features use the `lazyflow.operator.Operator` interface and are intended for
use in a `deeplearning.workflow.Workflow`.
"""

from .window import OpRawWindowed
from .window import OpDiff
from .window import OpFairness

from .filter import OpMean
from .filter import OpLinearWeightedMean
from .filter import OpExponentialFilter
from .filter import OpGaussianSmoothing

from .recent import OpRecent

from .combiners import OpSimpleCombiner
from .combiners import OpChain
