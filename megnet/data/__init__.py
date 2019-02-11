import abc


class Featurizer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def featurize(self, val):
        return val
