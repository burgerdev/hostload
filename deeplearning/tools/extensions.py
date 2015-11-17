
from pylearn2.train_extensions import TrainExtension
from deeplearning.tools import Buildable


class WeightKeeper(TrainExtension, Buildable):
    """
    a TrainExtension to keep track of the model's weights at each monitor step
    """
    @classmethod
    def build(cls, config, workingdir=None):
        return cls()

    def on_monitor(self, model, dataset, algorithm):
        """
        save the model's weights
        """
        self.__weights.append(model.get_param_values())

    def setup(self, model, dataset, algorithm):
        """
        initialize the weight list
        """
        self.__weights = []

    def get_weights(self):
        return self.__weights
