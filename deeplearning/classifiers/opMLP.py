
import logging
from itertools import repeat

import numpy as np
import theano

from deeplearning.data import OpDataset
from deeplearning.tools import Classification
from deeplearning.tools import Regression

from .abcs import OpTrain
from .abcs import OpPredict

from .opDeep import getTerminationCriteria


from pylearn2.models import mlp
from pylearn2.training_algorithms import bgd
from pylearn2 import train
from pylearn2.train_extensions import best_params

logger = logging.getLogger(__name__)


class OpMLPTrain(OpTrain, Classification, Regression):
    @classmethod
    def build(cls, d, parent=None, graph=None, workingdir=None):
        """
        configuration:
            d["layer_classes"] = tuple or class
            d["layer_sizes"] = tuple or int
        """
        op = cls(parent=parent, graph=graph)

        op._layer_classes = d["layer_classes"]
        op._size_hidden_layers = d["layer_sizes"]

        return op

    def __init__(self, *args, **kwargs):
        super(OpMLPTrain, self).__init__(*args, **kwargs)

        self._opTrainData = OpDataset(parent=self)
        self._opTrainData.Input.connect(self.Train)

        self._opValidData = OpDataset(parent=self)
        self._opValidData.Input.connect(self.Valid)

    def setupOutputs(self):
        super(OpMLPTrain, self).setupOutputs()

        self._configureLayers()

    def execute(self, slot, subregion, roi, result):
        self._train()
        result[0] = self._nn

    def _configureLayers(self):
        nvis = self.Train[0].meta.shape[1]
        layers = []
        layer_sizes = self._getLayerSizeIterator()
        for i, cls in enumerate(self._layer_classes):            
            name = "hidden_{:02d}".format(i)
            config = {"layer_name": name}
            if isinstance(cls, dict):
                actual_class = cls["class"]
                config.update(cls)
                del config["class"]
            elif issubclass(cls, mlp.Layer):
                config["irange"] = .1
                config["dim"] = layer_sizes.next()
                actual_class = cls
            else:
                raise ValueError("invalid layer: {}".format(cls))

            layer = actual_class(**config)
            layers.append(layer)

        if len(layers) > 0:
            last_layer = layers[-1]
            if hasattr(last_layer, "dim"):
                last_dim = last_layer.dim
            elif hasattr(last_layer, "output_channels"):
                last_dim = last_layer.output_channels
            else:
                raise ValueError("don't know where the layer stores its dim")
        else:
            last_dim = nvis
        n_out = self.Train[1].meta.shape[1]

        if n_out == 1:
            # set up for regression
            output = mlp.Linear(n_out, 'output', irange=.1)
            self._regression = True
        else:
            # set up for classification
            output = mlp.Softmax(n_out, 'output', irange=.1)
            self._regression = False
        layers.append(output)
        self._nn = mlp.MLP(layers=layers, nvis=nvis)

        if self._regression:
            # try to initialize weights smartly
            w = np.ones((last_dim, 1), dtype=np.float32)/last_dim
            output.set_weights(w)

    def _train(self):
        logger.info("============ TRAINING SUPERVISED ============")
        ds = self._opTrainData
        vds = self._opValidData

        if self._regression:
            channel = "valid_objective"
        else:
            channel = "valid_output_misclass"

        tc = getTerminationCriteria(epochs=200, channel=channel)
        keep = best_params.MonitorBasedSaveBest(
            channel_name=channel, store_best_model=True)

        trainer = bgd.BGD(line_search_mode='exhaustive',
                          batch_size=1000,
                          monitoring_dataset={'valid': vds},
                          monitoring_batch_size=1000,
                          termination_criterion=tc,
                          seed=None)

        nn = self._nn

        t = train.Train(dataset=ds, model=nn,
                        algorithm=trainer,
                        extensions=[keep])
        t.main_loop()

        # set best parameters to layer
        params = keep.best_model.get_param_values()
        nn.set_param_values(params)
        best_cost = keep.best_cost
        logger.info("Restoring model with cost {}".format(best_cost))

    def _getLayerSizeIterator(self):
        try:
            i = iter(self._size_hidden_layers)
        except TypeError:
            # layer size is a single integer
            i = repeat(self._size_hidden_layers)

        while True:
            yield i.next()


class OpMLPPredict(OpPredict, Classification, Regression):
    def execute(self, slot, subregion, roi, result):
        model = self.Classifier.value

        a = roi.start[0]
        b = roi.stop[0]

        inputs = self.Input[a:b, ...].wait()
        inputs = inputs.astype(np.float32)
        shared = theano.shared(inputs, name='inputs')
        prediction = model.fprop(shared).eval()

        a = roi.start[1]
        b = roi.stop[1]
        result[...] = prediction[:, a:b]
