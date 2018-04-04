from abc import ABCMeta, abstractmethod

class Attack(metaclass=ABCMeta):
    @abstractmethod
    def run(self, x, y, target):
        '''
        Returns an adversarial example for original input `x` and true label
        `y`. If `target` is not `None`, then the adversarial example should be
        targeted to be classified as `target`.
        '''
        raise NotImplementedError
