
import logging
from itertools import repeat

from .abcs import OpTrain

from deeplearning.data import OpDataset
from deeplearning.tools import Classification

from pylearn2.models import mlp
from pylearn2.models import rbm
from pylearn2.training_algorithms import sgd
from pylearn2.training_algorithms import bgd
from pylearn2.training_algorithms import learning_rule
from pylearn2.train_extensions import best_params
from pylearn2.energy_functions import rbm_energy
from pylearn2 import termination_criteria
from pylearn2 import blocks
from pylearn2 import corruption
from pylearn2 import train
from pylearn2.costs import ebm_estimation
from pylearn2.datasets import transformer_dataset


logger = logging.getLogger(__name__)


class OpDeepTrain(OpTrain, Classification):
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
            """
            layer = rbm.GaussianBinaryRBM(rbm_energy.grbm_type_1(),
                                          nvis=nvis, nhid=nhid,
                                          irange=.1,
                                          learn_sigma=True,
                                          init_sigma=.1)
            """
            layer = rbm.RBM(nvis=nvis, nhid=nhid,
                            irange=4, init_bias_hid=4,
                            init_bias_vis=0,
                            monitor_reconstruction=True)
            nvis = nhid
            layers.append(layer)

        n_out = self.Train[1].meta.shape[1]
        output = mlp.Softmax(n_out, 'output', irange=.1)
        layers.append(output)
        self._layers = layers

    def _pretrain(self):
        # corruptor = corruption.GaussianCorruptor(stdev=.05)
        # cost = ebm_estimation.SMD(corruptor=corruptor)
        cost = ebm_estimation.CDk(1)
        # cost = ebm_estimation.SML(10, 5)

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

            channel = "train_reconstruction_error"

            lr = learning_rule.Momentum(init_momentum=.5)
            lra = sgd.MonitorBasedLRAdjuster(channel_name=channel)
            keep =  best_params.MonitorBasedSaveBest(
                channel_name=channel, store_best_model=True)
            ext = [lra, keep]

            epochs = 200
            tc = self._getTerminationCriteria(epochs=epochs, channel=channel)

            trainer = sgd.SGD(learning_rate=.05, batch_size=50,
                              learning_rule=lr,
                              termination_criterion=tc,
                              monitoring_dataset={'train': tds},
                              monitor_iteration_mode="sequential",
                              monitoring_batch_size=1000,
                              cost=cost,
                              seed=None,
                              train_iteration_mode='sequential')

            t = train.Train(dataset=tds, model=layer,
                            algorithm=trainer,
                            extensions=ext)
            t.main_loop()

            # set best parameters to layer
            params = keep.best_model.get_param_values()
            layer.set_param_values(params)
            best_cost = keep.best_cost
            logger.info("Restoring model with cost {}".format(best_cost))

            tds = getTransform(self._layers[:i+1])

    def _trainAll(self):
        logger.info("============ TRAINING SUPERVISED ============")
        nvis = self.Train[0].meta.shape[1]
        ds = self._opTrainData
        vds = self._opValidData

        channel = "valid_output_misclass"
        epochs = 200

        tc = self._getTerminationCriteria(epochs=epochs, channel=channel)

        trainer = bgd.BGD(line_search_mode='exhaustive',
                          batch_size=1000,
                          monitoring_dataset={'valid': vds},
                          monitoring_batch_size=1000,
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

    def _getTerminationCriteria(self, epochs=None, channel=None):
        tc = []
        if epochs is not None:
            tc.append(termination_criteria.EpochCounter(epochs))
        if channel is not None:
            tc.append(termination_criteria.MonitorBased(
                channel_name=channel, prop_decrease=.00, N=20))
        return termination_criteria.And(tc)

    def _getLayerSizeIterator(self):
        try:
            i = iter(self._size_hidden_layers)
        except TypeError:
            # layer size is a single integer
            i = repeat(self._size_hidden_layers)

        while True:
            yield i.next()

    def _isPretrainable(self, layer):
        return isinstance(layer, rbm.RBM)
