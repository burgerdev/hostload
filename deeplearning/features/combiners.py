

from lazyflow.operator import Operator
from lazyflow.operator import InputSlot
from lazyflow.operator import OutputSlot
from lazyflow.operators.generic import OpMultiArrayStacker
from lazyflow.operators import OpReorderAxes

from deeplearning.tools import Buildable
from deeplearning.tools import build_operator


class OpSimpleCombiner(Operator, Buildable):
    """
    combines a list of feature operators into one (horizontally)

    operators must have slots Input and Output
    """
    Input = InputSlot()
    Output = OutputSlot()

    @classmethod
    def build(cls, config, parent=None, graph=None, workingdir=None):
        """
        config["operators"] = <tuple of operator classes or config dicts>
        """
        to_combine = config["operators"]
        operator = cls(to_combine, parent=parent, graph=graph)
        return operator

    def __init__(self, to_combine, *args, **kwargs):
        super(OpSimpleCombiner, self).__init__(*args, **kwargs)
        combiner = OpMultiArrayStacker(parent=self)
        combiner.AxisFlag.setValue('c')

        operators = [build_operator(item, parent=self) for item in to_combine]
        combiner.Images.resize(len(operators))
        for index, operator in enumerate(operators):
            combiner.Images[index].connect(operator.Output)
            operator.Input.connect(self.Input)
        self._combiner = combiner
        self._operators = operators
        self.Output.connect(combiner.Output)

    def propagateDirty(self, slot, subindex, roi):
        # is propagated internally
        pass


class OpChain(Operator, Buildable):
    """
    chains a list of feature operators (vertically)

    operators must have slots Input and Output
    """
    Input = InputSlot()
    Output = OutputSlot()

    @classmethod
    def build(cls, config, parent=None, graph=None, workingdir=None):
        """
        config["operators"] = <tuple of operator classes or config dicts>
        """
        to_combine = config["operators"]
        operator = cls(to_combine, parent=parent, graph=graph)
        return operator

    def __init__(self, to_combine, *args, **kwargs):
        super(OpChain, self).__init__(*args, **kwargs)
        #reorder = OpReorderAxes(parent=self)
        #reorder.AxisOrder.setValue('tc')
        #reorder.Input.connect(self.Input)
        #next_slot = reorder.Output
        next_slot = self.Input

        operators = [build_operator(item, parent=self) for item in to_combine]

        for operator in operators:
            operator.Input.connect(next_slot)
            next_slot = operator.Output
        self.Output.connect(next_slot)
        self.operators = operators

    def propagateDirty(self, slot, subindex, roi):
        # is propagated internally
        pass
