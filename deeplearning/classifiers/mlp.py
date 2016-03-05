
import logging
import os
import cPickle as pkl

import numpy as np
import theano

from itertools import repeat

from deeplearning.data import OpDataset
from deeplearning.tools import build_operator
from deeplearning.tools import Classification
from deeplearning.tools import Regression
from deeplearning.tools import IncompatibleDataset
from deeplearning.tools.extensions import PersistentTrainExtension
from deeplearning.tools.extensions import MonitorBasedSaveBest

from .abcs import OpTrain
from .abcs import OpPredict

from .deep import get_termination_criteria
from .deep import get_layer_size_iterator

from .mlp_init import ModelWeightInitializer
from .mlp_init import LayerWeightInitializer


from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.training_algorithms import learning_rule
from pylearn2 import train

logger = logging.getLogger(__name__)


class OpMLPTrain(OpTrain, Classification, Regression):
    @classmethod
    def get_default_config(cls):
        config = OpTrain.get_default_config()
        config["layer_sizes"] = (20, 10)
        config["layer_classes"] = (mlp.Sigmoid, mlp.Sigmoid)
        config["weight_initializer"] = {"class": ModelWeightInitializer}
        config["learning_rate"] = .1
        config["max_epochs"] = 40
        config["terminate_early"] = True
        config["init_momentum"] = .5
        config["batch_size"] = 100
        config["monitor_batch_size"] = 1000
        config["extensions"] = tuple()
        config["continue_learning"] = False
        return config

    @classmethod
    def build(cls, config, parent=None, graph=None, workingdir=None):
        obj = super(OpMLPTrain, cls).build(config, parent=parent, graph=graph,
                                           workingdir=workingdir)
        obj._workingdir = workingdir
        return obj

    def __init__(self, *args, **kwargs):
        super(OpMLPTrain, self).__init__(*args, **kwargs)

        self._opTrainData = OpDataset(parent=self)
        self._opTrainData.Input.connect(self.Train)

        self._opValidData = OpDataset(parent=self)
        self._opValidData.Input.connect(self.Valid)

        self._is_configured = False

    def setupOutputs(self):
        super(OpMLPTrain, self).setupOutputs()

        self._sanity_checks()

        if self._reconfigure():
            self._configure_layers()

    def execute(self, slot, subregion, roi, result):
        self._train()
        result[0] = self._nn

    def get_num_dimensions(self):
        """
        how many dimensions are we expecting the input to have

        override for e.g. CNN subclass
        """
        return 2

    def _reconfigure(self):
        #TODO perfrom checks if config is still valid
        return not self._is_configured

    def _configure_layers(self):
        nvis = self.Train[0].meta.shape[1]
        layers = []
        layer_sizes = get_layer_size_iterator(self._layer_sizes)
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

        self._initialize_weights()

        self._is_configured = True

    def _initialize_weights(self):
        use_initializers = True
        if self._continue_learning:
            #TODO this should probably done somewhere else
            filename = os.path.join(self._workingdir, "..", "classifierCache",
                                    "OpPickleCache.pkl")
            if not os.path.exists(filename):
                logger.warn("can't find trained model, starting from scratch")
            else:
                use_initializers = False
                with open(filename, 'r') as f:
                    model = pkl.load(f)
                if isinstance(model, np.ndarray):
                    model = model[0]
                params = model.get_param_values()
                self._nn.set_param_values(params)
                logger.info("restored model from {}".format(filename))

        if use_initializers:
            self._initialize_weights_from_initializers()

    def _initialize_weights_from_initializers(self):
        if isinstance(self._weight_initializer, LayerWeightInitializer):
            logger.warn("old-style initializer in config")
            config = {"initializers": repeat(self._weight_initializer)}
            init = ModelWeightInitializer.build(config, parent=self)
        elif isinstance(self._weight_initializer, tuple):
            logger.warn("old-style initializer tuple in config")
            config = {"initializers": self._weight_initializer}
            init = ModelWeightInitializer.build(config, parent=self)
        elif isinstance(self._weight_initializer, ModelWeightInitializer):
            init = self._weight_initializer
        else:
            msg = {"weight_initializer": self._weight_initializer}
            raise ValueError("unknown config entry: {}".format(msg))

        init.Data.connect(self._opTrainData.Output[0])
        init.Target.connect(self._opTrainData.Output[1])
        init.init_model(self._nn)

    def _train(self):
        logger.info("============ TRAINING SUPERVISED ============")
        tds = self._opTrainData
        vds = self._opValidData

        if self._regression:
            # channel = "valid_objective"
            channel = "train_objective"
        else:
            channel = "valid_output_misclass"

        ext = []
        if channel is not None:
            keep = MonitorBasedSaveBest.build(dict(channel=channel))
            ext.append(keep)
            lra = sgd.MonitorBasedLRAdjuster(channel_name=channel)
            ext.append(lra)

        for other in self._extensions:
            ext.append(build_operator(other, workingdir=self._workingdir))

        self.extensions_used = ext

        if self._terminate_early:
            termination_channel = channel
        else:
            termination_channel = None
        criteria = get_termination_criteria(epochs=self._max_epochs,
                                            channel=termination_channel)

        monitors = {'train': tds, 'valid': vds}

        algorithm = sgd.SGD(learning_rate=self._learning_rate,
                            batch_size=self._batch_size,
                            learning_rule=learning_rule.Momentum(
                                init_momentum=self._init_momentum),
                            termination_criterion=criteria,
                            monitoring_dataset=monitors,
                            monitor_iteration_mode="sequential",
                            monitoring_batch_size=self._monitor_batch_size,
                            seed=None,
                            train_iteration_mode='sequential')

        trainer = train.Train(dataset=tds, model=self._nn,
                              algorithm=algorithm,
                              extensions=ext)
        trainer.main_loop()

        # set best parameters to layer
        params = keep.best_params
        best_cost = keep.best_cost
        logger.info("Restoring model with cost {}".format(best_cost))
        self._nn.set_param_values(params)

        for ext in self.extensions_used:
            if isinstance(ext, PersistentTrainExtension):
                ext.store()

    def _sanity_checks(self):
        for slot in (self.Train, self.Valid):
            ext = " ({})".format(slot.name)
            assert len(slot) == 2, "need features AND prediction" + ext
            assert slot[0].meta.shape[0] == slot[1].meta.shape[0],\
                "#features and #predictions must agree" + ext
            if len(slot[0].meta.shape) != self.get_num_dimensions():
                msg = "{} is not {}-D".format(slot[0].meta.shape,
                                              self.get_num_dimensions())
                raise IncompatibleDataset(msg)


class OpMLPPredict(OpPredict, Classification, Regression):
    def execute(self, slot, subregion, roi, result):
        model = self.Classifier.value
        if isinstance(model, np.ndarray):
            model = model[0]

        a = roi.start[0]
        b = roi.stop[0]

        inputs = self.Input[a:b, ...].wait()
        inputs = inputs.astype(np.float32)
        shared = theano.shared(inputs, name='inputs')
        prediction = model.fprop(shared).eval()

        a = roi.start[1]
        b = roi.stop[1]
        result[...] = prediction[:, a:b]
