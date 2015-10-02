
from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.operators import OpReorderAxes

from .converters import OpDiscretize
from .segmented import OpExponentiallySegmentedPattern

from deeplearning.tools import Buildable


class OpHostloadTarget(Operator, Buildable):
    Input = InputSlot()
    WindowSize = InputSlot()
    NumLevels = InputSlot()
    Output = OutputSlot()

    @classmethod
    def build(cls, d, parent=None, graph=None, workingdir=None):
        op = cls(parent=parent, graph=graph)
        op.NumLevels.setValue(d["num_levels"])
        op.WindowSize.setValue(d["window_size"])
        return op

    def __init__(self, *args, **kwargs):
        super(OpHostloadTarget, self).__init__(*args, **kwargs)
        self._drop = OpReorderAxes(parent=self)
        self._drop.AxisOrder.setValue('t')

        self._opExp = OpExponentiallySegmentedPattern(parent=self)
        self._opExp.NumSegments.setValue(1)
        self._opExp.BaselineSize.connect(self.WindowSize)

        self._opDisc = OpDiscretize(parent=self)
        self._opDisc.NumLevels.connect(self.NumLevels)

        self.Output.connect(self._opDisc.Output)
        self._opDisc.Input.connect(self._opExp.Output)
        self._opExp.Input.connect(self._drop.Output)
        self._drop.Input.connect(self.Input)

    def setupOutputs(self):
        pass

    def propagateDirty(self, slot, subindex, roi):
        pass

    def execute(self, slot, subindex, roi, result):
        # can't be reached because of internal connections
        raise NotImplementedError()
