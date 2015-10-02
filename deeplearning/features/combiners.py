

from lazyflow.operator import Operator
from lazyflow.operator import InputSlot
from lazyflow.operator import OutputSlot
from lazyflow.operators.generic import OpMultiArrayStacker

from deeplearning.tools import Buildable


class OpSimpleCombiner(Operator, Buildable):
    """
    combines a list of feature operators into one
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

        def createOperator(item):
            """
            create an operator from an item in the config dict
            """
            if isinstance(item, dict):
                operator = item["class"].build(item, parent=self)
            elif issubclass(item, Operator):
                operator = item(parent=self)
            else:
                raise ValueError("cannot construct operator from {}"
                                 "".format(type(item)))
            return operator

        operators = [createOperator(item) for item in to_combine]
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
