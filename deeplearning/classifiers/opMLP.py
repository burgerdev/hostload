
import logging
from itertools import repeat

import numpy as np
import theano

from deeplearning.data import OpDataset
from deeplearning.tools import Classification
from deeplearning.tools import Regression

from .abcs import OpTrain
from .abcs import OpPredict


from pylearn2.models import mlp
from pylearn2.training_algorithms import bgd
from pylearn2 import termination_criteria

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
            nhid = layer_sizes.next()
            layer = cls(dim=nhid, irange=.1, layer_name=name)
            layers.append(layer)

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
            w = np.ones((nhid, 1), dtype=np.float32)/nhid
            output.set_weights(w)

    def _train(self):
        logger.info("============ TRAINING SUPERVISED ============")
        ds = self._opTrainData
        vds = self._opValidData

        if self._regression:
            channel = "valid_objective"
        else:
            channel = "valid_output_misclass"

        tc_a = termination_criteria.EpochCounter(500)
        tc_b = termination_criteria.MonitorBased(
            channel_name=channel,
            prop_decrease=.00, N=20)
        tc = termination_criteria.And((tc_a, tc_b))

        trainer = bgd.BGD(line_search_mode='exhaustive',
                          batch_size=1000,
                          monitoring_dataset={'valid': vds},
                          monitoring_batch_size=1000,
                          termination_criterion=tc,
                          seed=None)

        nn = self._nn

        trainer.setup(nn, ds)
        while True:
            trainer.train(dataset=ds)
            nn.monitor.report_epoch()
            nn.monitor()
            if not trainer.continue_learning(nn):
                break

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
