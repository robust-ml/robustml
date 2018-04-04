from abc import ABCMeta, abstractmethod

class Model(metaclass=ABCMeta):
    '''
    Interface for a model (classifier).

    Besides the required methods below, a model should do a reasonable job of
    providing easy access to internals to make white box attacks easier. For
    example, a model using TensorFlow might want to provide access to the input
    tensor placeholder and the tensor representing the logits output of the
    classifier.
    '''

    @property
    @abstractmethod
    def dataset(self):
        '''
        A concrete instance of a subclass of `robustml.dataset.Dataset`.
        '''
        raise NotImplementedError

    @property
    @abstractmethod
    def threat_model(self):
        '''
        An instance of `robustml.threat_model.ThreatModel`, ideally
        one of the pre-defined concrete threat models.
        '''
        raise NotImplementedError

    @abstractmethod
    def classify(self, x):
        '''
        Returns the label for the input x (as a Python integer).
        '''
        raise NotImplementedError
