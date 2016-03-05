"""
deep Boltzmann machines (greedy unsupervised pretraining + supervised training)
"""

import logging
from itertools import repeat

from .abcs import OpTrain

from deeplearning.data import OpDataset
from deeplearning.tools import Classification

from lazyflow.operator import InputSlot

from pylearn2.models import mlp
from pylearn2.models import rbm
from pylearn2.training_algorithms import sgd
from pylearn2.training_algorithms import bgd
from pylearn2.training_algorithms import learning_rule
from pylearn2.train_extensions import best_params
from pylearn2 import termination_criteria
from pylearn2 import blocks
from pylearn2 import train
from pylearn2.costs import ebm_estimation
from pylearn2.datasets import transformer_dataset


LOGGER = logging.getLogger(__name__)


class OpDeepTrain(OpTrain, Classification):
    """
    create an MLP and train it with the DBN approach

    (Classifier is compatible with OpMLPPredict)
    """

    NumHiddenLayers = InputSlot()
    SizeHiddenLayers = InputSlot()

    _dbn = None
    _layers = []

    @classmethod
    def build(cls, d, parent=None, graph=None, workingdir=None):
        """
        configuration needs:
            d["num_hidden_layers"] = <int>
            d["size_hidden_layers"] = <int>
        """
        operator = cls(parent=parent, graph=graph)

        operator.NumHiddenLayers.setValue(d["num_hidden_layers"])
        operator.SizeHiddenLayers.setValue(d["size_hidden_layers"])

        return operator

    def __init__(self, *args, **kwargs):
        super(OpDeepTrain, self).__init__(*args, **kwargs)

        self._train_data = OpDataset(parent=self)
        self._train_data.Input.connect(self.Train)

        self._valid_data = OpDataset(parent=self)
        self._valid_data.Input.connect(self.Valid)

    def setupOutputs(self):
        super(OpDeepTrain, self).setupOutputs()

        self._configure_layers()

    def execute(self, slot, _, roi, result):
        self._pretrain()
        self._train_all()
        result[0] = self._dbn

    def setInSlot(self, slot, subindex, roi, value):
        raise NotImplementedError()

    def _configure_layers(self):
        """
        create all the RBM layers according to configuration
        """
        nvis = self.Train[0].meta.shape[1]
        layers = []
        layer_sizes = get_layer_size_iterator(self.SizeHiddenLayers.value)
        for _ in range(self.NumHiddenLayers.value):
            nhid = layer_sizes.next()
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
        """
        run greedy pretraining on all RBM layers
        """
        cost = ebm_estimation.CDk(1)
        # cost = ebm_estimation.SML(10, 5)

        dataset = self._train_data

        def get_transform(layers):
            """
            closure for mapping the original dataset through pretrained layers
            """
            transformed_dataset = transformer_dataset.TransformerDataset(
                raw=dataset,
                transformer=blocks.StackedBlocks(layers=layers))
            return transformed_dataset

        transformed_dataset = dataset
        for i, layer in enumerate(self._layers):
            if not self._is_pretrainable(layer):
                continue

            LOGGER.info("============ TRAINING UNSUPERVISED ============")
            LOGGER.info("============        Layer %d        ============", i)

            channel = "train_reconstruction_error"

            lra = sgd.MonitorBasedLRAdjuster(channel_name=channel)
            keep = best_params.MonitorBasedSaveBest(
                channel_name=channel, store_best_model=True)
            ext = [lra, keep]

            criteria = get_termination_criteria(epochs=200, channel=channel)

            algorithm = sgd.SGD(learning_rate=.05, batch_size=50,
                                learning_rule=learning_rule.Momentum(
                                    init_momentum=.5),
                                termination_criterion=criteria,
                                monitoring_dataset={'train':
                                                    transformed_dataset},
                                monitor_iteration_mode="sequential",
                                monitoring_batch_size=1000,
                                cost=cost,
                                seed=None,
                                train_iteration_mode='sequential')

            trainer = train.Train(dataset=transformed_dataset, model=layer,
                                  algorithm=algorithm,
                                  extensions=ext)
            trainer.main_loop()

            # set best parameters to layer
            layer.set_param_values(keep.best_model.get_param_values())
            LOGGER.info("Restoring model with cost %f", keep.best_cost)

            # redefinition of transformed_dataset is ok
            # pylint: disable=R0204
            transformed_dataset = get_transform(self._layers[:i+1])
            # pylint: enable=R0204

    def _train_all(self):
        """
        supervised training (after unsupervised pretraining)
        """
        LOGGER.info("============ TRAINING SUPERVISED ============")
        nvis = self.Train[0].meta.shape[1]
        tds = self._train_data
        vds = self._valid_data

        channel = "valid_output_misclass"

        criteria = get_termination_criteria(epochs=200, channel=channel)

        algorithm = bgd.BGD(line_search_mode='exhaustive',
                            batch_size=1000,
                            monitoring_dataset={'valid': vds},
                            monitoring_batch_size=1000,
                            termination_criterion=criteria,
                            seed=None)

        layers = self._layers

        def layer_mapping(index, layer):
            """
            tell MLP that RBM layers are pretrained
            """
            if not self._is_pretrainable(layer):
                return layer
            name = "{}_{:02d}".format(layer.__class__.__name__, index)
            return mlp.PretrainedLayer(layer_name=name,
                                       layer_content=layer)
        layers = [layer_mapping(index, layer)
                  for index, layer in enumerate(layers)]

        dbn = mlp.MLP(layers, nvis=nvis)

        trainer = train.Train(dataset=tds, model=dbn,
                              algorithm=algorithm,
                              extensions=[])
        trainer.main_loop()

        self._dbn = dbn

    @staticmethod
    def _is_pretrainable(layer):
        """
        can this layer be pretrained?
        """
        return isinstance(layer, rbm.RBM)


def get_layer_size_iterator(int_or_iterable):
    """
    get an iterator for different inputs:
        iterable -> iterate over iterable
        other type -> repeat this type ad infinitum
    """
    try:
        iter_ = iter(int_or_iterable)
    except TypeError:
        # layer size is a single integer
        iter_ = repeat(int_or_iterable)

    while True:
        yield iter_.next()


def get_termination_criteria(epochs=None, channel=None):
    """
    construct AND'ed termination criteria from
        * max number of training epochs
        * non-decrease in some monitored channel
    """
    criteria = []
    if epochs is not None:
        criteria.append(termination_criteria.EpochCounter(epochs))
    if channel is not None:
        criteria.append(termination_criteria.MonitorBased(
            channel_name=channel, prop_decrease=.00, N=20))
    return termination_criteria.And(criteria)
