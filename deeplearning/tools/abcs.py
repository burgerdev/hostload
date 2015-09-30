
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
