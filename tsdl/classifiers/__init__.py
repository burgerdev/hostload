"""
classifier operators (training operators and prediction operators)
"""

from .svm import OpSVMTrain
from .svm import OpSVMPredict
from .state import OpStateTrain
from .state import OpStateTrainRegression
from .state import OpStatePredict
from .state import OpStatePredictRegression
from .deep import OpDeepTrain
from .mlp import OpMLPTrain
from .mlp import OpMLPPredict
from .rf import OpRFTrain
from .rf import OpRFPredict
