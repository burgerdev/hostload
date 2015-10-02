
import logging

from lazyflow.operator import Operator

LOGGER = logging.getLogger(__name__)


class Buildable(object):
    """
    mix-in that provides standard build() support for lazyflow.Operator
    """
    @classmethod
    def build(cls, config, parent=None, graph=None, workingdir=None):
        """
        build an instance of this class with given configuration dict
        """
        operator = cls(parent=parent, graph=graph)
        return operator


def buildOperator(data, **kwargs):
    """
    construct an operator, either from a config dict or from the bare class

    Inputs:
        data: either config dict with at least a key "class"
              or a lazyflow.operator.Operator subclass
        kwargs: keyword args passed to Buildable.build
    Output:
        adequate Operator instance
    """
    if isinstance(data, dict):
        assert "class" in data, "need to know the Operator's class"
        cls = data["class"]
        if not issubclass(cls, Buildable):
            assert hasattr(cls, "build"), "can't build class {}".format(cls)
            LOGGER.warn(
                "{} is not Buildable but has a build method".format(cls))
        return cls.build(data, **kwargs)
    else:
        if "workingdir" in kwargs:
            del kwargs["workingdir"]
        assert issubclass(data, Operator), "{} is no Operator".format(data)
        return data(**kwargs)
