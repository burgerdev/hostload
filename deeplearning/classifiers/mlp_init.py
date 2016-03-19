
from itertools import izip, repeat

import numpy as np
import vigra
import theano

from lazyflow.operator import Operator, InputSlot, OutputSlot


from deeplearning.tools import Buildable
from deeplearning.tools import get_rng
from deeplearning.tools import build_operator

try:
    from scipy import cluster
except ImportError:
    _HAVE_SCIPY = False
else:
    _HAVE_SCIPY = True


class ModelWeightInitializer(Operator, Buildable):
    Data = InputSlot()
    Target = InputSlot()

    def setupOutputs(self):
        pass

    def propagateDirty(self, slot, subindex, roi):
        pass

    def init_model(self, model):
        sub_inits = self._initializers
        if isinstance(sub_inits, LayerWeightInitializer):
            sub_inits = repeat(sub_inits)

        last_dim = model.get_input_space().dim
        visited_layers = []
        for init, layer in izip(sub_inits, model.layers):
            if isinstance(init, dict):
                init = build_operator(init, parent=self)
            next_dim = layer.get_output_space().dim
            if isinstance(init, OperatorLayerWeightInitializer):
                forward = OpForwardLayers(visited_layers, parent=self)
                forward.Input.connect(self.Data)
                init.Input.resize(2)
                init.Input[1].connect(self.Target)
                init.Input[0].connect(forward.Output)

            init.init_layer(layer, nvis=last_dim, nhid=next_dim)
            last_dim = next_dim
            visited_layers.append(layer)

    @classmethod
    def get_default_config(cls):
        config = super(ModelWeightInitializer, cls).get_default_config()
        config["rng"] = get_rng()
        config["initializers"] = {"class": StandardWeightInitializer}
        return config


class LayerWeightInitializer(Buildable):
    def __init__(self, *args, **kwargs):
        pass

    def init_layer(self, layer, nvis=1, nhid=1):
        raise NotImplementedError()

    @classmethod
    def get_default_config(cls):
        config = super(LayerWeightInitializer, cls).get_default_config()
        config["rng"] = get_rng()
        return config


class StandardWeightInitializer(LayerWeightInitializer):
    @classmethod
    def get_default_config(cls):
        config = super(StandardWeightInitializer, cls).get_default_config()
        config["irange"] = 0.1
        return config

    def init_layer(self, layer, nvis=1, nhid=1):
        weights = (self._rng.rand(nvis, nhid) - .5)*(2*self._irange)
        weights = weights.astype(np.float32)
        layer.set_weights(weights)
        biases = np.zeros((nhid,), dtype=np.float32)
        biases[:] = 0
        layer.set_biases(biases)


class NormalWeightInitializer(LayerWeightInitializer):
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


class OperatorLayerWeightInitializer(Operator, LayerWeightInitializer):
    Input = InputSlot(level=1)

    def setupOutputs(self):
        pass

    def propagateDirty(self, slot, subindex, roi):
        pass


class PCAWeightInitializer(OperatorLayerWeightInitializer):
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


class LeastSquaresWeightInitializer(OperatorLayerWeightInitializer):
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
        forwarded = np.asarray(shared.eval())
        result[:] = forwarded[..., roi.start[1]:roi.stop[1]]


class OptimalInitializer(ModelWeightInitializer):
    """
    optimal initializer (according to approximation theory) for Sigmoid layers
    """

    def __init__(self, *args, **kwargs):
        assert _HAVE_SCIPY, "need scipy for using this initializer"
        super(OptimalInitializer, self).__init__(*args, **kwargs)

    def init_model(self, model):
        if len(model.layers) != 3:
            raise ValueError("currently only 3-layer models supported")

        #TODO check if model really is sigmoid

        n_vis = model.get_input_space().dim
        n_centroids = model.layers[1].get_output_space().dim
        n_intermediate = model.layers[0].get_output_space().dim
        if n_intermediate != 2*n_vis*n_centroids:
            raise ValueError("wrong number of neurons in layer 0")

        centroids = self._get_cluster_centroids(n_centroids)

        weights, biases = self._get_weights(centroids)

        for weight, bias, layer in zip(weights, biases, model.layers):
            weight = weight.astype(np.float32)
            layer.set_weights(weight)
            bias = bias.astype(np.float32)
            layer.set_biases(bias)

    @classmethod
    def get_default_config(cls):
        config = super(OptimalInitializer, cls).get_default_config()
        config["rng"] = get_rng()
        config["num_clusters"] = 5
        del config["initializers"]
        return config

    def _get_cluster_centroids(self, n_centroids):
        data = self.Data[...].wait()
        target = self.Target[...].wait()
        concat = np.concatenate((data, target), axis=1)
        centroids, _ = cluster.vq.kmeans(concat, n_centroids)
        return centroids

    @classmethod
    def _get_weights(cls, centroids):
        """
        last axis of centroids should be target value
        """
        scale = 5  # FIXME
        normalization = 0.2449  # (1-np.exp(-.5))/(1+np.exp(-.5))

        weights = []
        biases = []

        n_vis = centroids.shape[1] - 1
        n_centroids = centroids.shape[0]
        n_intermediate = 2*n_vis*n_centroids

        # first layer: bump functions
        weights0 = np.zeros((n_vis, n_intermediate))
        biases0 = np.zeros((n_intermediate,))

        # second layer: sigma of bump
        weights1 = np.zeros((n_intermediate, n_centroids))
        biases1 = np.ones((n_centroids,))*scale*(.5 - n_vis)

        for i in range(n_centroids):
            centroid = centroids[i, :-1]
            eye = scale*np.eye(n_vis)
            weights0[:, 2*i*n_vis:(2*i+1)*n_vis] = eye
            weights0[:, (2*i+1)*n_vis:(2*i+2)*n_vis] = eye
            biases0[2*i*n_vis:(2*i+1)*n_vis] = .5 - scale*centroid
            biases0[(2*i+1)*n_vis:(2*i+2)*n_vis] = -.5 - scale*centroid

            weights1[2*i*n_vis:(2*i+1)*n_vis, i] = scale/normalization
            weights1[(2*i+1)*n_vis:(2*i+2)*n_vis, i] = -scale/normalization

        weights.append(weights0)
        weights.append(weights1)
        biases.append(biases0)
        biases.append(biases1)

        # last layer: weighting of inputs
        weights.append(centroids[:, -1:])
        biases.append(np.zeros(1,))

        return weights, biases


class GridInitializer(ModelWeightInitializer):
    """
    rasterized initialization
    """

    def __init__(self, *args, **kwargs):
        super(GridInitializer, self).__init__(*args, **kwargs)

    @classmethod
    def get_default_config(cls):
        config = super(GridInitializer, cls).get_default_config()
        config["rng"] = get_rng()
        del config["initializers"]
        return config

    def init_model(self, model):
        self._model = model

        self._prepare()

        dim_input = model.get_input_space().dim
        dim_layer1 = model.layers[0].get_output_space().dim
        dim_layer2 = model.layers[1].get_output_space().dim

        num_bins = dim_layer1 // dim_input

        weights, biases = self._get_first_layer(dim_input, num_bins)
        self._model.layers[0].set_weights(weights.astype(np.float32))
        self._model.layers[0].set_biases(biases.astype(np.float32))

        num_bumps = dim_layer2
        weights, biases = self._get_second_layer(dim_input, num_bins,
                                                 num_bumps)
        self._model.layers[0].set_weights(weights.astype(np.float32))
        self._model.layers[0].set_biases(biases.astype(np.float32))

    def _get_first_layer(self, dim_input, num_bins):
        # divide each input dimension [0,1] into bins
        bin_edges = np.linspace(0, 1, num_bins+1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.0
        # center the bins
        scale = float(num_bins)
        biases_1d = -scale*bin_centers

        weights = np.zeros((dim_input, self._dims[1]))
        biases = np.zeros((self._dims[1],))

        for dim in range(dim_input):
            weights[dim, dim*num_bins:(dim+1)*num_bins] = scale

        biases = np.concatenate((biases_1d,)*dim_input)
        return weights, biases

    def _get_second_layer(self, dim_input, num_bins, num_bumps):
        weights = np.zeros((dim_input*num_bins, num_bumps))
        biases = np.zeros((num_bumps,))

        return weights, biases

    def _prepare(self):
        model = self._model
        dims = np.zeros((len(model.layers)+1,), dtype=np.int)
        dims[0] = model.get_input_space().dim
        for i, layer in enumerate(model.layers):
            dims[i+1] = model.layers[i].get_output_space().dim
        self._dims = dims

        check(len(model.layers) == 3,
              "initializer works with exactly 3 layers")
        check(self._dims[1] % self._dims[0] == 0,
              "size of first layer must be a multiple of input dimensionality")


def check(cond, msg):
    if not cond:
        raise PreconditionError(msg)


class PreconditionError(Exception):
    def __init__(self, msg, *args, **kwargs):
        msg2 = "Precondition not met: {}".format(str(msg))
        super().__init__(self, msg2, *args, **kwargs)


if __name__ == "__main__":
    from lazyflow.graph import Graph
    from pylearn2.models.mlp import MLP
    from pylearn2.models.mlp import Sigmoid
    from pylearn2.models.mlp import Linear

    num_dim = 2
    num_bins = 3

    x = np.asarray([0, 0, 1, 1, .5])
    y = np.asarray([0, 1, 0, 1, .5])
    vol = np.zeros((5, num_dim), dtype=np.float32)
    vol[:, 0] = x
    vol[:, 1] = y
    vol = vigra.taggedView(vol, axistags='tc')
    target = np.zeros((vol.shape[0], 1))
    target = vigra.taggedView(target, axistags='tc')

    """
    layers = [Sigmoid(layer_name='bumps', irange=0, dim=2*nvis*ncent),
              Sigmoid(layer_name='cents', irange=0, dim=ncent),
              Linear(layer_name='out', irange=0, dim=1)]
    """

    layers = [Sigmoid(layer_name='1d', irange=0, dim=num_dim*num_bins),
              Linear(layer_name='1d', irange=0, dim=num_dim*num_bins)]

    mlp = MLP(layers=layers, nvis=num_dim)

    init = GridInitializer.build({}, graph=Graph())
    init.Data.setValue(vol)
    init.Target.setValue(target)
    init.init_model(mlp)

    op = OpForwardLayers(layers[:], graph=Graph())
    op.Input.setValue(vol)

    z_pred = op.Output[...].wait().squeeze()
    for i, sub in enumerate(z_pred):
        print("=======")
        print(i)
        print(vol[i])
        print(sub.reshape((2, 3)))