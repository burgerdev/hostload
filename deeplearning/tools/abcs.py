
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
        default_config = cls.get_default_config()
        for key in config:
            assert key in default_config,\
                "invalid config entry '{}'".format(key)

        default_config.update(config)

        operator = cls(parent=parent, graph=graph)

        operator._set_attributes(default_config)

        return operator

    @classmethod
    def get_default_config(cls):
        """
        override to provide your own default configuration
        """
        return {"class": cls.__name__}

    def _set_attributes(self, config):
        """
        dynamically set attributes from config dict

        automatically builds child operators from config dicts found
        """
        for key in config:
            if key == "class":
                continue
            entry = config[key]
            if isinstance(entry, dict) and "class" in entry:
                entry = build_operator(entry, parent=self)
            attr = "_" + key
            setattr(self, attr, entry)


def build_operator(data, **kwargs):
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
