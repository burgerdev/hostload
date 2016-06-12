"""
base classes that define training and prediction operator interfaces
"""

import numpy as np

from tsdl.tools import Operator, InputSlot, OutputSlot
from tsdl.tools import SubRegion

from tsdl.tools import Buildable


class OpTrain(Operator, Buildable):
    """
    lazyflow operator that trains a model

    inputs are training and validation data:
        first subslot contains features
        second subslot contains targets

    Output must be a single element np.array that contains a model which can
    be used with the appropriate OpPredict.

    Don't forget to use the mix-ins
        tsdl.tools.[Classification|Regression]
    to indicate the supported types of training.
    """

    Train = InputSlot(level=1)
    Valid = InputSlot(level=1)

    Classifier = OutputSlot()

    def setupOutputs(self):
        self.Classifier.meta.shape = (1,)
        self.Classifier.meta.dtype = np.object

    def propagateDirty(self, slot, subindex, roi):
        self.Classifier.setDirty(slice(None))

    def execute(self, slot, subindex, roi, result):
        raise NotImplementedError()


class OpPredict(Operator, Buildable):
    """
    lazyflow operator that predicts from a trained model

    Chain this operator to OpTrain.

    Input receives features.
    Target receives the ground truth (must only be used for shape checking)

    Output should have the same shape and dtype as Target.

    Don't forget to use the mix-ins
        tsdl.tools.[Classification|Regression]
    to indicate the supported types of prediction.
    """

    Input = InputSlot()
    Classifier = InputSlot()
    Target = InputSlot()

    Output = OutputSlot()

    def setupOutputs(self):
        num_examples = self.Input.meta.shape[0]
        num_channels = self.Target.meta.shape[1]
        self.Output.meta.shape = (num_examples, num_channels)
        self.Output.meta.dtype = np.float

    def propagateDirty(self, slot, subindex, roi):
        if slot == self.Classifier:
            self.Output.setDirty(slice(None))
        elif slot == self.Input:
            new_roi = SubRegion(self.Output,
                                start=(roi.start[0], 0),
                                stop=(roi.stop[0], self.Target.meta.shape[1]))
            self.Output.setDirty(new_roi)
        # don't need to handle self.Target because we need it just for
        # Target.meta.shape

    def execute(self, slot, subindex, roi, result):
        raise NotImplementedError()
