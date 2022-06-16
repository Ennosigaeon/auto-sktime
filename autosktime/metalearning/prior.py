import abc
from typing import Union

import numpy as np
import pandas as pd
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, Hyperparameter, NormalFloatHyperparameter, \
    NormalIntegerHyperparameter, BetaFloatHyperparameter, BetaIntegerHyperparameter, NumericalHyperparameter
from matplotlib import pyplot as plt
from scipy.stats import truncnorm, norm
from statsmodels.nonparametric.kde import KDEUnivariate

from ConfigSpace import UniformIntegerHyperparameter, OrdinalHyperparameter, CategoricalHyperparameter, Constant


class Prior(abc.ABC):

    def __init__(self, hp: Hyperparameter):
        self.hp = hp

    @abc.abstractmethod
    def calculate(self, vector: np.ndarray):
        pass


class UniformPrior(Prior):

    def calculate(self, vector: np.ndarray):
        if isinstance(self.hp, BetaFloatHyperparameter) or isinstance(self.hp, BetaIntegerHyperparameter):
            return np.ones_like(vector, dtype=np.float64)
        if isinstance(self.hp, UniformFloatHyperparameter) or isinstance(self.hp, UniformIntegerHyperparameter):
            ub = 1
            lb = 0
            inside_range = ((lb <= vector) & (vector <= ub)).astype(int)
            return inside_range / (self.hp.upper - self.hp.lower)
        elif isinstance(self.hp, NormalFloatHyperparameter) or isinstance(self.hp, NormalIntegerHyperparameter):
            return np.ones_like(vector, dtype=np.float64)
        elif isinstance(self.hp, OrdinalHyperparameter) or isinstance(self.hp, CategoricalHyperparameter):
            return np.ones_like(vector, dtype=np.float64) / self.hp.get_size()
        elif isinstance(self.hp, Constant):
            return np.ones_like(vector, dtype=np.float64)
        else:
            raise ValueError(f'Hyperparameter {type(self.hp)} not supported yet')


class NormalPrior(Prior):

    def __init__(self, hp: Hyperparameter, mu: float, sigma: float):
        super().__init__(hp)
        self.mu = mu
        self.sigma = sigma

    def calculate(self, vector: np.ndarray):
        if getattr(self.hp, 'lower', None) is not None and getattr(self.hp, 'upper', None) is not None:
            # noinspection PyUnresolvedReferences
            a = (self.hp.lower - self.mu) / self.sigma
            # noinspection PyUnresolvedReferences
            b = (self.hp.upper - self.mu) / self.sigma
            return truncnorm(a, b, loc=self.mu, scale=self.sigma).pdf(self.hp._transform(vector))
        elif isinstance(self.hp, NormalFloatHyperparameter) or isinstance(self.hp, NormalIntegerHyperparameter):
            return norm(loc=self.mu, scale=self.sigma).pdf(self.hp._transform(vector))
        elif isinstance(self.hp, Constant):
            return np.ones_like(vector, dtype=np.float64)
        else:
            raise ValueError(f'Hyperparameter {type(self.hp)} not supported yet')


class KdePrior(Prior):

    def __init__(
            self,
            hp: Hyperparameter,
            observations: pd.Series,
            weights: pd.Series = None,
            bw: Union[str, float, int] = 'silverman'
    ):
        super().__init__(hp)
        self.observations = observations.dropna()
        self._fit(weights, bw)

    def _fit(self, weights: pd.Series, bw: Union[str, float, int], plot: bool = False):
        if isinstance(self.hp, OrdinalHyperparameter) or isinstance(self.hp, CategoricalHyperparameter):
            if weights is None:
                weights = pd.Series(np.ones_like(self.observations) / len(self.observations),
                                    index=self.observations.index)

            choices = self.hp.sequence if hasattr(self.hp, 'sequence') else self.hp.choices
            unique, counts = np.unique(self.observations, return_counts=True)
            self.weights_ = pd.Series(
                {
                    **{key: 0 for key in choices},  # Set all values to 0
                    **{key: ((self.observations == key).astype(int) * weights).sum()
                       for key in unique}  # Fill with actual counts
                })
        elif isinstance(self.hp, NumericalHyperparameter):
            self.kde_ = KDEUnivariate(self.observations.values)
            if weights is not None:
                weights = weights.values
            self.kde_.fit(bw=bw, weights=weights, fft=weights is None)

            if plot:
                fig = plt.figure(figsize=(12, 5))
                ax = fig.add_subplot(111)
                ax.plot(self.kde_.support, self.kde_.density)
                plt.show()

    def calculate(self, vector: np.ndarray):
        if isinstance(self.hp, Constant):
            return np.ones_like(vector, dtype=np.float64)
        elif isinstance(self.hp, OrdinalHyperparameter) or isinstance(self.hp, CategoricalHyperparameter):
            values = [self.hp._transform(int(s)) for s in vector]
            return [self.weights_[s] for s in values]
        elif isinstance(self.hp, NumericalHyperparameter):
            return np.array([self.kde_.evaluate(xi) for xi in vector]).flatten()
        else:
            raise ValueError(f'Hyperparameter {type(self.hp)} not supported yet')
