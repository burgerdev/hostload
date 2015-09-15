
import logging
from itertools import repeat

from .abcs import OpTrain

from deeplearning.data import OpDataset

from pylearn2.models import mlp
from pylearn2.models import rbm
from pylearn2.training_algorithms import sgd
from pylearn2.training_algorithms import bgd
from pylearn2.energy_functions import rbm_energy
from pylearn2 import termination_criteria
from pylearn2 import blocks
from pylearn2 import corruption
from pylearn2.costs import ebm_estimation
from pylearn2.datasets import transformer_dataset


logger = logging.getLogger(__name__)


class OpDeepTrain(OpTrain):
    @classmethod
    def build(cls, d, parent=None, graph=None, workingdir=None):
        """
        configuration needs:
            d["num_hidden_layers"] = <int>
            d["size_hidden_layers"] = <int>
        """
        op = cls(parent=parent, graph=graph)

        op._num_hidden_layers = d["num_hidden_layers"]
        op._size_hidden_layers = d["size_hidden_layers"]

        return op

    def __init__(self, *args, **kwargs):
        super(OpDeepTrain, self).__init__(*args, **kwargs)

        self._opTrainData = OpDataset(parent=self)
        self._opTrainData.Input.connect(self.Train)

        self._opValidData = OpDataset(parent=self)
        self._opValidData.Input.connect(self.Valid)

    def setupOutputs(self):
        super(OpDeepTrain, self).setupOutputs()

        self._configureLayers()

    def execute(self, slot, subregion, roi, result):
        self._pretrain()
        self._trainAll()
        result[0] = self._nn

    def _configureLayers(self):
        nvis = self.Train[0].meta.shape[1]
        layers = []
        layer_sizes = self._getLayerSizeIterator()
        for i in range(self._num_hidden_layers):
            nhid = layer_sizes.next()
            layer = rbm.GaussianBinaryRBM(rbm_energy.grbm_type_1(),
                                          nvis=nvis, nhid=nhid,
                                          irange=.1)
            nvis = nhid
            layers.append(layer)

        n_out = self.Train[1].meta.shape[1]
        output = mlp.Softmax(n_out, 'output', irange=.1)
        layers.append(output)
        self._layers = layers

    def _pretrain(self):
        corruptor = corruption.GaussianCorruptor(stdev=.4)
        cost = ebm_estimation.SMD(corruptor=corruptor)

        ds = self._opTrainData

        def getTransform(layers):
            tds = transformer_dataset.TransformerDataset(
                raw=ds,
                transformer=blocks.StackedBlocks(layers=layers))
            return tds

        tds = ds
        for i, layer in enumerate(self._layers):
            if not self._isPretrainable(layer):
                continue

            logger.info("============ TRAINING UNSUPERVISED ============")
            logger.info("============        Layer {}        ============"
                        "".format(i))

            tc_a = termination_criteria.EpochCounter(500)
            tc_b = termination_criteria.MonitorBased(
                channel_name="train_objective", prop_decrease=.01, N=20)
            tc = termination_criteria.And((tc_a, tc_b))
            trainer = sgd.SGD(learning_rate=.05, batch_size=10,
                              termination_criterion=tc,
                              monitoring_dataset={'train': tds},
                              cost=cost,
                              seed=None,
                              train_iteration_mode='sequential')
            trainer.setup(layer, tds)
            while True:
                trainer.train(dataset=tds)
                layer.monitor.report_epoch()
                layer.monitor()
                if not trainer.continue_learning(layer):
                    break

            tds = getTransform(self._layers[:i+1])

    def _trainAll(self):
        logger.info("============ TRAINING SUPERVISED ============")
        nvis = self.Train[0].meta.shape[1]
        ds = self._opTrainData
        vds = self._opValidData

        tc_a = termination_criteria.EpochCounter(500)
        tc_b = termination_criteria.MonitorBased(
            channel_name="valid_objective", prop_decrease=.00, N=20)
        tc = termination_criteria.And((tc_a, tc_b))

        trainer = bgd.BGD(line_search_mode='exhaustive',
                          batch_size=5000,
                          monitoring_dataset={'valid': vds},
                          termination_criterion=tc,
                          seed=None)

        layers = self._layers

        def layerMapping((index, layer)):
            if not self._isPretrainable(layer):
                return layer
            name = "{}_{:02d}".format(layer.__class__.__name__, index)
            return mlp.PretrainedLayer(layer_name=name,
                                       layer_content=layer)
        layers = map(layerMapping, enumerate(layers))

        nn = mlp.MLP(layers, nvis=nvis)

        trainer.setup(nn, ds)
        while True:
            trainer.train(dataset=ds)
            nn.monitor.report_epoch()
            nn.monitor()
            if not trainer.continue_learning(nn):
                break
        self._nn = nn

    def _getLayerSizeIterator(self):
        try:
            i = iter(self._size_hidden_layers)
        except TypeError:
            # layer size is a single integer
            i = repeat(self._size_hidden_layers)

        while True:
            yield i.next()

    def _isPretrainable(self, layer):
        return isinstance(layer, rbm.GaussianBinaryRBM)
