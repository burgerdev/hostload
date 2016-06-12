
import vigra

from matplotlib import pyplot as plt

from lazyflow.graph import Graph

from tsdl.targets import OpExponentiallySegmentedPattern


def run(args):
    x = vigra.readHDF5(args.h5file, args.internal_path)
    t = x[:, 0]/float(24*60*60*1e6)
    data = vigra.taggedView(x[:, 1], axistags='t')

    op = OpExponentiallySegmentedPattern(graph=Graph())
    op.Input.setValue(data)
    op.BaselineSize.setValue(60)
    op.NumSegments.setValue(6)

    out = op.Output[...].wait()
    leg = ["mean cpu usage over {} hours".format(2**i)
           for i in range(out.shape[1])]
    plt.plot(t, out)
    plt.legend(leg)
    plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("h5file", help="cpu usage hdf5 file")
    parser.add_argument("internal_path",
                        help="cpu usage hdf5 file internal path")

    args = parser.parse_args()
    run(args)
