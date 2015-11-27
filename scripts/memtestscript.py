
import numpy as np


from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.training_algorithms import learning_rule
from pylearn2 import train
from pylearn2.train_extensions import best_params
from pylearn2 import termination_criteria
from pylearn2.datasets import DenseDesignMatrix


def test_pylearn2(num_epochs):
    num_features = 32
    num_examples = 100000
    cut = int(.9*num_examples)
    X = np.random.rand(num_examples, num_features)
    Y = np.power(np.prod(X, axis=1), 1.0/num_features)

    tds = DenseDesignMatrix(X=X[:cut, :], y=Y[:cut, np.newaxis])
    vds = DenseDesignMatrix(X=X[cut:, :], y=Y[cut:, np.newaxis])

    layers = [mlp.Sigmoid(dim=24, irange=.1, layer_name="a"),
              mlp.Sigmoid(dim=16, irange=.1, layer_name="b"),
              mlp.Linear(dim=1, irange=.1, layer_name="c")]

    model = mlp.MLP(layers=layers, nvis=num_features)

    channel = "valid_objective"
    extensions = [sgd.MonitorBasedLRAdjuster(channel_name=channel),
                  best_params.MonitorBasedSaveBest(channel_name=channel,
                                                   store_best_model=True)]

    termination = termination_criteria.EpochCounter(num_epochs)

    algorithm = sgd.SGD(learning_rate=0.2,
                        batch_size=100,
                        learning_rule=learning_rule.Momentum(
                            init_momentum=0.5),
                        termination_criterion=termination,
                        monitoring_dataset={'valid': vds},
                        monitor_iteration_mode="sequential",
                        monitoring_batch_size=1000,
                        train_iteration_mode='sequential')

    trainer = train.Train(dataset=tds, model=model,
                          algorithm=algorithm,
                          extensions=extensions)
    trainer.main_loop()


def test_lazyflow(num_epochs):
    pass


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-n", "--epochs", type=int, default=10000,
                        help="number of epochs to test")
    parser.add_argument("-p", "--pylearn2", action="store_true",
                        default=False,
                        help="run plain pylearn2 test")
    parser.add_argument("-l", "--lazyflow", action="store_true",
                        default=False,
                        help="run plain lazyflow test")

    args = parser.parse_args()

    if args.pylearn2:
        test_pylearn2(args.epochs)

    if args.lazyflow:
        test_lazyflow(args.epochs)
