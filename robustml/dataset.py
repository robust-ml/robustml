from abc import ABCMeta, abstractmethod

'''
Identifiers for known datasets.
'''

class Dataset:
    '''
    You should not subclass this in your own code, you should only use the
    datasets defined here.
    '''

    @property
    @abstractmethod
    def shape(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def labels(self):
        raise NotImplementedError

class MNIST(Dataset):
    '''
    Data points are 28x28 arrays with elements in [0, 1].
    '''

    @property
    def shape(self):
        return (28, 28)

    @property
    def labels(self):
        return 10

class CIFAR10(Dataset):
    '''
    Data points are 32x32x3 arrays with elements in [0, 1].
    '''

    @property
    def shape(self):
        return (32, 32, 3)

    @property
    def labels(self):
        return 10

class ImageNet(Dataset):
    '''
    Data points are ?x?x3 arrays with elements in [0, 1].

    Dimensions are specified in the constructor.
    '''

    def __init__(self, shape=None):
        '''
        Shape is a 3-tuple (height, width, channels) describing the shape of
        the input image to the model.
        '''
        if not isinstance(shape, tuple) or len(shape) != 3 \
                or not all(isinstance(i, int) for i in shape) \
                or not shape[-1] == 3:
            raise ValueError('bad shape: %s' % str(shape))
        self._shape = shape
    
    @property
    def shape(self):
        return self._shape

    @property
    def labels(self):
        return 1000
