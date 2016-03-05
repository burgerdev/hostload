
import signal

from .extensions import BuildableTrainExtension


class SignalExtension(BuildableTrainExtension):
    """
    tell the pylearn2.train.Train object to stop when SIGHUP is received

    This class needs to be imported from the main thread because it registers
    a UNIX signal handler.
    """
    signaled = False

    def on_monitor(self, model, dataset, algorithm):
        """
        request quit if a signal was received
        """
        if self.signaled:
            raise StopIteration()

    @classmethod
    def handler(cls, signum, _):
        if signum in (signal.SIGHUP, signal.SIGTERM):
            cls.signaled = True
        else:
            raise RuntimeError("caught signal {}".format(signum))


signal.signal(signal.SIGHUP, SignalExtension.handler)
