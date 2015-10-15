
import logging

import numpy as np
import theano

from itertools import repeat

from lazyflow.operator import Operator, InputSlot

from deeplearning.data import OpDataset
from deeplearning.tools import Buildable
from deeplearning.tools import build_operator
from deeplearning.tools import Classification
from deeplearning.tools import Regression
from deeplearning.tools import IncompatibleDataset

from .abcs import OpTrain
from .abcs import OpPredict

from .deep import get_termination_criteria
from .deep import get_layer_size_iterator


from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.training_algorithms import learning_rule
from pylearn2 import train
from pylearn2.train_extensions import best_params

logger = logging.getLogger(__name__)


class OpMLPTrain(OpTrain, Classification, Regression):
    @classmethod
    def get_default_config(cls):
        config = OpTrain.get_default_config()
        config["layer_sizes"] = (20, 10)
        config["layer_classes"] = (mlp.Sigmoid, mlp.Sigmoid)
        config["weight_initializer"] = {"class": WeightInitializer}
        config["learning_rate"] = .1
        config["max_epochs"] = 40
        config["terminate_early"] = True
        config["init_momentum"] = .5
        config["batch_size"] = 100
        config["monitor_batch_size"] = 1000
        return config

    def __init__(self, *args, **kwargs):
        super(OpMLPTrain, self).__init__(*args, **kwargs)

        self._opTrainData = OpDataset(parent=self)
        self._opTrainData.Input.connect(self.Train)

        self._opValidData = OpDataset(parent=self)
        self._opValidData.Input.connect(self.Valid)

    def setupOutputs(self):
        super(OpMLPTrain, self).setupOutputs()

        self._sanity_checks()

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

    def _initialize_weights(self):
        if isinstance(self._weight_initializer, WeightInitializer):
            iter_ = repeat(self._weight_initializer)
        else:
            iter_ = [build_operator(config, parent=self)
                     for config in self._weight_initializer]

        last_dim = self._nn.get_input_space().dim
        for init, layer in zip(iter_, self._nn.layers):
            next_dim = layer.get_output_space().dim
            if hasattr(init, "Input"):
                init.Input.connect(self._opTrainData.Output)

            init.init_layer(layer, nvis=last_dim, nhid=next_dim)
            last_dim = next_dim

    def _train(self):
        logger.info("============ TRAINING SUPERVISED ============")
        tds = self._opTrainData
        vds = self._opValidData

        if self._regression:
            channel = "valid_objective"
        else:
            channel = "valid_output_misclass"
        if not self._terminate_early:
            channel = None

        lra = sgd.MonitorBasedLRAdjuster(channel_name=channel)
        ext = [lra]
        if channel is not None:
            keep = best_params.MonitorBasedSaveBest(
                channel_name=channel, store_best_model=True)
            ext.append(keep)

        criteria = get_termination_criteria(epochs=self._max_epochs,
                                            channel=channel)

        algorithm = sgd.SGD(learning_rate=self._learning_rate,
                            batch_size=self._batch_size,
                            learning_rule=learning_rule.Momentum(
                                init_momentum=self._init_momentum),
                            termination_criterion=criteria,
                            monitoring_dataset={'valid': vds},
                            monitor_iteration_mode="sequential",
                            monitoring_batch_size=self._monitor_batch_size,
                            seed=None,
                            train_iteration_mode='sequential')

        trainer = train.Train(dataset=tds, model=self._nn,
                              algorithm=algorithm,
                              extensions=ext)
        trainer.main_loop()

        # set best parameters to layer
        params = keep.best_model.get_param_values()
        self._nn.set_param_values(params)
        best_cost = keep.best_cost
        logger.info("Restoring model with cost {}".format(best_cost))

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




class WeightInitializer(Buildable):
    def __init__(self, *args, **kwargs):
        pass

    def init_layer(self, layer, nvis=1, nhid=1):
        raise NotImplementedError()


class NormalWeightInitializer(WeightInitializer):
    @classmethod
    def get_default_config(cls):
        config = super(WeightInitializer, cls).get_default_config()
        config["mean"] = 0
        config["stddev"] = .1
        config["bias"] = .5
        return config

    def init_layer(self, layer, nvis=1, nhid=1):
        weights = np.random.normal(self._mean, self._stddev, size=(nvis, nhid))
        weights = weights.astype(np.float32)
        layer.set_weights(weights)
        biases = np.zeros((nhid,), dtype=np.float32)
        biases[:] = self._bias
        layer.set_biases(biases)


class FilterWeightInitializer(WeightInitializer):
    @classmethod
    def get_default_config(cls):
        config = super(WeightInitializer, cls).get_default_config()
        return config

    def init_layer(self, layer, nvis=1, nhid=1):
        weights = np.ones((nvis, nhid))
        weights /= nvis
        signs = np.random.randint(0, 3, size=weights.shape)
        weights *= (signs-1)
        weights = weights.astype(np.float32)
        layer.set_weights(weights)
        biases = np.random.random(size=(nhid,)).astype(np.float32)
        layer.set_biases(biases)


class _OperatorWeightInitializer(Operator, WeightInitializer):
    Input = InputSlot(level=1)

    def setupOutputs(self):
        pass

    def propagateDirty(self, slot, subindex, roi):
        pass


class PCAWeightInitializer(_OperatorWeightInitializer):
    def init_layer(self, layer, nvis=1, nhid=1):
        assert self.Input[0].ready(), "need dataset to compute PCA"
        X = self.Input[0][...].wait()
        weights, biases = self._pca(X, num_components=nhid)
        weights = weights.astype(np.float32)
        layer.set_weights(weights)
        biases = biases.astype(np.float32)
        layer.set_biases(biases)

    def _pca(self, X, num_components=1):
        X_mean = X.mean(axis=0)
        X_centered = X - X_mean[np.newaxis, :]
    
        A = np.dot(X_centered.T, X_centered)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        idx = np.argsort(eigenvalues)
        idx = np.flipud(idx)
        assert num_components <= len(idx)
        idx = idx[:num_components]
        eigenvectors = eigenvectors[:, idx]
        constants = -np.dot(X_mean, eigenvectors)
        return eigenvectors, constants


class LeastSquaresWeightInitializer(_OperatorWeightInitializer):
    def init_layer(self, layer, nvis=1, nhid=1):
        assert self.Input.ready(), "need dataset to compute PCA"
        X = self.Input[0][...].wait()
        y = self.Input[1][...].wait().squeeze()
        weights, biases = self._solve(X, y, num_components=nhid)
        weights = weights.astype(np.float32)
        layer.set_weights(weights)
        biases = biases.astype(np.float32)
        layer.set_biases(biases)

    def _solve(self, X, y, num_components=1):
        dim = X.shape[1]
        #print("dim={}".format(dim))
        assert num_components <= dim
        assert len(X) == len(y) and len(y.shape) == 1

        # generate random hyperplanes (A,b) that intersect [0,1]^dim at P
        num_hyperplanes = min(num_components*10, len(X)/4)
        # FIXME line below is biased
        A = np.random.random(size=(num_hyperplanes, dim)) -.5
        A_norms = np.sqrt(np.square(A).sum(axis=1, keepdims=True))
        A /= A_norms
        P = np.random.random(size=(num_hyperplanes, dim))
        b = -(A*P).sum(axis=1)

        def sigmoid(x):
            return 1/(1+np.exp(-x))

        # determine output weights by least squares fit
        print("computing matrix")
        M = sigmoid(np.dot(X, A.T) + b)
        
        print("done, matrix dims are {}".format(M.shape))
        c = np.dot(np.linalg.pinv(M), y)

        # take hyperplanes corresponding to large weights
        importance = np.flipud(np.argsort(np.abs(c)))[:num_components]
        A = A[importance, :].T
        b = b[importance]
        assert len(b) == num_components
        assert A.shape == (dim, num_components)
        return A, b


class OpMLPPredict(OpPredict, Classification, Regression):
    def execute(self, slot, subregion, roi, result):
        model = self.Classifier.value

        a = roi.start[0]
        b = roi.stop[0]

        inputs = self.Input[a:b, ...].wait()
        # TODO check if tests work without this statement
        # inputs = inputs.astype(np.float32)
        shared = theano.shared(inputs, name='inputs')
        prediction = model.fprop(shared).eval()

        a = roi.start[1]
        b = roi.stop[1]
        result[...] = prediction[:, a:b]
