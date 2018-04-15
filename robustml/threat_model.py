from abc import ABCMeta, abstractmethod
import numpy as np

from . import dataset

class ThreatModel(metaclass=ABCMeta):
    @abstractmethod
    def check(self, original, perturbed):
        '''
        Returns whether the perturbed image is a valid perturbation of the
        original under the threat model.

        `original` and `perturbed` are numpy arrays of the same dtype and
        shape.
        '''
        raise NotImplementedError

    @property
    @abstractmethod
    def targeted(self):
        '''
        Returns whether the threat model only includes targeted attacks
        (requiring the attack to be capable of synthesizing targeted
        adversarial examples).
        '''
        raise NotImplementedError

class Or(ThreatModel):
    '''
    A union of threat models.
    '''

    def __init__(self, *threat_models):
        self._threat_models = threat_models

    def check(self, original, perturbed):
        return any(i.check(original, perturbed) for i in self._threat_models)

    @property
    def targeted(self):
        return all(i.targeted for i in self._threat_models)

class And(ThreatModel):
    '''
    An intersection of threat models.
    '''

    def __init__(self, *threat_models):
        self._threat_models = threat_models

    def check(self, original, perturbed):
        return all(i.check(original, perturbed) for i in self._threat_models)

    @property
    def targeted(self):
        return any(i.targeted for i in self._threat_models)

class Lp(ThreatModel):
    '''
    Bounded L_p perturbation. Given a `p` and `epsilon`, x' is a valid
    perturbation of x if the following holds:

    || x - x' ||_p <= \epsilon
    '''

    _SLOP = 0.0001 # to account for rounding errors

    def __init__(self, p, epsilon, targeted=False):
        self._p = p
        self._epsilon = epsilon
        self._targeted = targeted

    def check(self, original, perturbed):
        # we want to treat the inputs as big vectors
        original = np.ndarray.flatten(original)
        perturbed = np.ndarray.flatten(perturbed)
        # ensure it's a valid image
        if np.min(perturbed) < -self._SLOP or np.max(perturbed) > 1+self._SLOP:
            return False
        norm = np.linalg.norm(original - perturbed, ord=self._p)
        return norm <= self._epsilon + self._SLOP

    @property
    def targeted(self):
        return self._targeted

    @property
    def p(self):
        return self._p

    @property
    def epsilon(self):
        return self._epsilon

class L0(Lp):
    def __init__(self, epsilon, targeted=False):
        super().__init__(p=0, epsilon=epsilon, targeted=targeted)

class L1(Lp):
    def __init__(self, epsilon, targeted=False):
        super().__init__(p=1, epsilon=epsilon, targeted=targeted)

class L2(Lp):
    def __init__(self, epsilon, targeted=False):
        super().__init__(p=2, epsilon=epsilon, targeted=targeted)

class Linf(Lp):
    '''
    Bounded L_inf perturbation. Given a `p` and `epsilon`, x' is a valid
    perturbation of x if the following holds:

    || x - x' ||_\infty <= \epsilon

    >>> model = Linf(0.1)
    >>> x = np.array([0.1, 0.2, 0.3])
    >>> model.check(x, x)
    True
    >>> model.targeted
    False
    >>> model = Linf(0.1, targeted=True)
    >>> model.targeted
    True
    >>> y = np.array([0.1, 0.25, 0.32])
    >>> model.check(x, y)
    True
    >>> z = np.array([0.3, 0.2, 0.3])
    >>> model.check(x, z)
    False
    '''

    def __init__(self, epsilon, targeted=False):
        super().__init__(p=np.inf, epsilon=epsilon, targeted=targeted)
