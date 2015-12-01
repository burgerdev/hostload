
import logging
import os
import cPickle as pkl

import numpy as np
import vigra
import theano

from itertools import repeat

from lazyflow.operator import Operator, InputSlot, OutputSlot

from deeplearning.data import OpDataset
from deeplearning.tools import Buildable
from deeplearning.tools import get_rng
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
        config["weight_initializer"] = {"class": NormalWeightInitializer}
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
        if isinstance(self._weight_initializer, WeightInitializer):
            iter_ = repeat(self._weight_initializer)
        else:
            iter_ = [build_operator(config, parent=self)
                     for config in self._weight_initializer]

        last_dim = self._nn.get_input_space().dim
        visited_layers = []
        for init, layer in zip(iter_, self._nn.layers):
            next_dim = layer.get_output_space().dim
            if isinstance(init, OperatorWeightInitializer):
                forward = OpForwardLayers(visited_layers, parent=self)
                forward.Input.connect(self._opTrainData.Output[0])
                init.Input.resize(2)
                init.Input[1].connect(self._opTrainData.Output[1])
                init.Input[0].connect(forward.Output)

            init.init_layer(layer, nvis=last_dim, nhid=next_dim)
            last_dim = next_dim
            visited_layers.append(layer)

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


class WeightInitializer(Buildable):
    def __init__(self, *args, **kwargs):
        pass

    def init_layer(self, layer, nvis=1, nhid=1):
        raise NotImplementedError()

    @classmethod
    def get_default_config(cls):
        config = super(WeightInitializer, cls).get_default_config()
        config["rng"] = get_rng()
        return config


class NormalWeightInitializer(WeightInitializer):
    @classmethod
    def get_default_config(cls):
        config = super(NormalWeightInitializer, cls).get_default_config()
        config["mean"] = 0.0
        config["stddev"] = .1
        config["bias"] = 0.0
        return config

    def init_layer(self, layer, nvis=1, nhid=1):
        weights = self._rng.normal(self._mean, self._stddev, size=(nvis, nhid))
        weights = weights.astype(np.float32)
        layer.set_weights(weights)
        biases = np.zeros((nhid,), dtype=np.float32)
        biases[:] = self._bias
        layer.set_biases(biases)


class FilterWeightInitializer(WeightInitializer):
    def init_layer(self, layer, nvis=1, nhid=1):
        weights = np.ones((nvis, nhid))
        weights /= nvis
        signs = self._rng.randint(0, 3, size=weights.shape)
        weights *= (signs-1)
        weights = weights.astype(np.float32)
        layer.set_weights(weights)
        biases = self._rng.rand(nhid).astype(np.float32)
        layer.set_biases(biases)


class OperatorWeightInitializer(Operator, WeightInitializer):
    Input = InputSlot(level=1)

    def setupOutputs(self):
        pass

    def propagateDirty(self, slot, subindex, roi):
        pass


class PCAWeightInitializer(OperatorWeightInitializer):
    def init_layer(self, layer, nvis=1, nhid=1):
        assert self.Input[0].ready(), "need dataset to compute PCA"
        X = self.Input[0][...].wait()
        weights, biases = self._get_weights(X, num_components=nhid)
        weights = weights.astype(np.float32)
        layer.set_weights(weights)
        biases = biases.astype(np.float32)
        layer.set_biases(biases)

    @staticmethod
    def _pca(data):
        """
        apply PCA on dataset

        output is sorted by standard deviation in descending order

        @param data an array of shape num_observations x num_features
        @return tuple(standard_deviations, eigenvectors, mean_observation)
        """
        num_obs = len(data)
        assert num_obs > 1, "can't apply PCA on dataset with only one point"
        data_mean = data.mean(axis=0)
        data_centered = data - data_mean[np.newaxis, :]

        covariance_matrix = np.dot(data_centered.T, data_centered)/(num_obs-1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # A should be positive-semidefinite, but sometimes numpy won't
        # acknowledge that and return something like -1e-7
        variances = np.where(eigenvalues > 0, eigenvalues, 0)
        order = np.flipud(np.argsort(variances))

        return variances[order], eigenvectors[:, order], data_mean

    def _get_weights(self, data, num_components=1):
        variances, vectors, mean = self._pca(data)

        cumulated_energy = np.cumsum(variances)
        total_energy = cumulated_energy[-1]

        # We don't want to initialize for directions with very small
        # deviations, so we restrict ourselves to some fraction p of the total
        # energy.
        # Note that \sigma^2_{cutoff} >= (1-p)*total_energy/(dim - cutoff + 1).
        #FIXME feature turned off for now, usefulness is questionable
        # fraction = 0.95
        # cutoff = np.where(cumulated_energy >= fraction*total_energy)[0][0]
        cutoff = len(variances) - 1

        dim = cutoff + 1
        variances = variances[:dim]
        vectors = vectors[:, :dim]

        # use different signs
        vectors_interleaved = np.zeros((vectors.shape[0], 2*dim),
                                       dtype=np.float32)
        vectors_interleaved[:, ::2] = vectors
        vectors_interleaved[:, 1::2] = -vectors

        # initialized output
        weights = np.zeros((vectors.shape[0], num_components),
                           dtype=np.float32)

        # array to remember which sigma is used for this weight
        indices = np.zeros((num_components,), dtype=np.int)
        base = min(num_components, 2*dim)
        weights[:, :base] = vectors_interleaved[:, :base]
        indices[:base:2] = np.arange(dim)[:(base//2 + (1 if base % 2 else 0))]
        indices[1:base:2] = np.arange(dim)[:base//2]

        if num_components > 2*dim:
            # normalize variances so that we can use them as probabilities
            importance = variances / (total_energy - cumulated_energy[cutoff])
            importance_interleaved = np.zeros((2*dim,), dtype=np.float32)
            importance_interleaved[::2] = importance
            importance_interleaved[1::2] = importance
            importance_interleaved /= 2

            # fill weights with eigenvectors weighted by their importance
            fill = num_components - 2*dim
            choice = self._rng.choice(len(importance_interleaved), size=fill,
                                      replace=True, p=importance_interleaved)
            weights[:, 2*dim:] = vectors_interleaved[:, choice]
            indices[2*dim:] = indices[choice]

            # add noise so that we don't have the same weight vector twice
            shape = (vectors.shape[0], fill)
            noise = (self._rng.rand(*shape) - .5) * 2
            noise /= np.sqrt(np.square(noise).sum(axis=0, keepdims=True))
            # noise is a random vector with 20% energy
            noise *= .2
            weights[:, 2*dim:] += noise

        # if we can, take all possible signs, otherwise flip sign randomly
        if num_components < dim:
            flip_sign_start = 0
        else:
            flip_sign_start = 2*dim

        if flip_sign_start < num_components:
            shape = (1, num_components - flip_sign_start)
            signs = self._rng.choice([-1, 1], size=shape)
            weights[:, flip_sign_start:] *= signs

        biases = -np.dot(mean, weights)

        # adjust weights and biases such that a vector of length 2*sigma scores
        # within [-2, 2]
        #FIXME this is only good for Sigmoid layers
        sigmas = np.sqrt(variances[indices])
        adjustment = sigmas
        weights /= adjustment[np.newaxis, :]
        biases /= adjustment
        return weights, biases


class LeastSquaresWeightInitializer(OperatorWeightInitializer):
    def init_layer(self, layer, nvis=1, nhid=1):
        assert self.Input.ready(), "need dataset to compute PCA"
        X = self.Input[0][...].wait()
        y = self.Input[1][...].wait().squeeze()
        self._layer = layer
        weights, biases = self._solve(X, y, num_components=nhid)
        weights = weights.astype(np.float32)
        layer.set_weights(weights)
        biases = biases.astype(np.float32)
        layer.set_biases(biases)
        self._layer = None

    def _solve(self, X, y, num_components=1):
        dim = X.shape[1]
        assert len(X) == len(y) and len(y.shape) == 1

        # generate random hyperplanes (A,b) that intersect [0,1]^dim at P

        # the equation system has to be overdetermined, i.e. we need more rows
        # (len(X)) than columns (num_hyperplanes)
        num_hyperplanes = min(num_components*10, len(X)/4)
        # FIXME line below is biased
        A = self._rng.rand(num_hyperplanes, dim) - .5
        A_norms = np.sqrt(np.square(A).sum(axis=1, keepdims=True))
        A /= A_norms
        P = self._rng.rand(num_hyperplanes, dim)
        b = -(A*P).sum(axis=1)

        # determine output weights by least squares fit
        M = self._apply_nonlinearity(np.dot(X, A.T) + b)
        c = np.dot(np.linalg.pinv(M), y)
        assert len(c) == num_hyperplanes

        # take hyperplanes corresponding to large weights
        importance = np.abs(c)
        importance /= importance.sum()
        choice = self._rng.choice(num_hyperplanes, size=num_components,
                                  p=importance)
        A = A[choice, :].T
        b = b[choice]
        assert len(b) == num_components
        assert A.shape == (dim, num_components)
        return A, b

    def _apply_nonlinearity(self, x):
        from pylearn2.models.mlp import Sigmoid, Linear, RectifiedLinear
        nonlinearities = {Sigmoid: lambda x: 1/(1+np.exp(-x)),
                          Linear: lambda x: x,
                          RectifiedLinear: lambda x: np.max(x, 0)}

        for cls in nonlinearities:
            if isinstance(self._layer, cls):
                return nonlinearities[cls](x)
        msg = "can't determine nonlinearity of class {}".format(
            type(self._layer))
        raise NotImplementedError(msg)


class OpMLPPredict(OpPredict, Classification, Regression):
    def execute(self, slot, subregion, roi, result):
        model = self.Classifier.value
        if isinstance(model, np.ndarray):
            model = model[0]

        a = roi.start[0]
        b = roi.stop[0]

        inputs = self.Input[a:b, ...].wait()
        # inputs = inputs.astype(np.float32)
        shared = theano.shared(inputs, name='inputs')
        prediction = model.fprop(shared).eval()

        a = roi.start[1]
        b = roi.stop[1]
        result[...] = prediction[:, a:b]


class OpForwardLayers(Operator):
    Input = InputSlot()
    Output = OutputSlot()

    def __init__(self, layers, *args, **kwargs):
        super(OpForwardLayers, self).__init__(*args, **kwargs)
        self._layers = layers

    def setupOutputs(self):
        if len(self._layers) > 0:
            dim_output = self._layers[-1].get_output_space().dim
        else:
            dim_output = self.Input.meta.shape[1]
        self._dim_output = dim_output
        num_inputs = self.Input.meta.shape[0]

        self.Output.meta.assignFrom(self.Input.meta)
        self.Output.meta.shape = (num_inputs, dim_output)
        self.Output.meta.axistags = vigra.defaultAxistags('tc')
        self.Output.meta.dtype = np.float32

    def propagateDirty(self, slot, subindex, roi):
        roi = roi.copy()
        roi.start = (roi.start[0], 0)
        roi.stop = (roi.stop[0], self._dim_output)
        self.Output.setDirty(roi)

    def execute(self, slot, subregion, roi, result):
        inputs = self.Input[roi.start[0]:roi.stop[0], ...].wait()
        inputs = inputs.astype(np.float32)
        shared = theano.shared(inputs)
        for layer in self._layers:
            shared = layer.fprop(shared)
        result[:] = shared.eval()
