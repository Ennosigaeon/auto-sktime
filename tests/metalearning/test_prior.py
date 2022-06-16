import unittest

import numpy as np
import pandas as pd
from ConfigSpace.hyperparameters import NormalFloatHyperparameter, BetaFloatHyperparameter, \
    NormalIntegerHyperparameter, BetaIntegerHyperparameter, CategoricalHyperparameter, OrdinalHyperparameter
from numpy.testing import assert_almost_equal
from scipy.stats import truncnorm, norm

from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant
from autosktime.metalearning.prior import UniformPrior, NormalPrior, KdePrior


class PriorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.values = np.linspace(0, 1, 100)

        self.uniform_float = UniformFloatHyperparameter('uniform_float', lower=-1, upper=2)
        self.normal_float = NormalFloatHyperparameter('normal_float', mu=1, sigma=0.1)
        self.beta_float = BetaFloatHyperparameter('beta_float', alpha=3, beta=2, lower=1, upper=4, log=False)

        self.uniform_integer = UniformIntegerHyperparameter('uniform_int', lower=-1, upper=2)
        self.normal_integer = NormalIntegerHyperparameter('normal_int', mu=1, sigma=2)
        self.beta_integer = BetaIntegerHyperparameter('beta_int', alpha=3, beta=2, lower=1, upper=4, log=False)

        self.constant = Constant('constant', 'value')
        self.categorical = CategoricalHyperparameter('cat', choices=['red', 'green', 'blue'])
        self.ordinal = OrdinalHyperparameter('ordinal', sequence=['10', '20', '30'])

    def test_uniform_prior(self):
        prior1 = UniformPrior(self.uniform_float).calculate(self.values)
        assert_almost_equal(prior1, np.ones(self.values.shape) * 1 / 3)

        prior2 = UniformPrior(self.normal_float).calculate(self.values)
        assert_almost_equal(prior2, np.ones(self.values.shape))

        prior3 = UniformPrior(self.beta_float).calculate(self.values)
        assert_almost_equal(prior3, np.ones(self.values.shape))

        prior4 = UniformPrior(self.uniform_integer).calculate(self.values)
        assert_almost_equal(prior4, np.ones(self.values.shape) * 1 / 3)

        prior5 = UniformPrior(self.normal_integer).calculate(self.values)
        assert_almost_equal(prior5, np.ones(self.values.shape))

        prior6 = UniformPrior(self.beta_integer).calculate(self.values)
        assert_almost_equal(prior6, np.ones(self.values.shape))

        prior7 = UniformPrior(self.constant).calculate(self.values)
        assert_almost_equal(prior7, np.ones(self.values.shape))

        prior8 = UniformPrior(self.categorical).calculate(self.values)
        assert_almost_equal(prior8, np.ones(self.values.shape) * 1 / 3)

        prior9 = UniformPrior(self.ordinal).calculate(self.values)
        assert_almost_equal(prior9, np.ones(self.values.shape) * 1 / 3)

    def test_normal_prior(self):
        prior1 = NormalPrior(self.uniform_float, mu=0, sigma=1).calculate(self.values)
        assert_almost_equal(prior1, truncnorm(-1, 2, loc=0, scale=1).pdf(self.uniform_float._transform(self.values)))

        prior2 = NormalPrior(self.normal_float, mu=0, sigma=1).calculate(self.values)
        assert_almost_equal(prior2, norm(loc=0, scale=1).pdf(self.normal_float._transform(self.values)))

        prior3 = NormalPrior(self.beta_float, mu=0, sigma=1).calculate(self.values)
        assert_almost_equal(prior3, truncnorm(1, 4, loc=0, scale=1).pdf(self.beta_float._transform(self.values)))

        prior4 = NormalPrior(self.uniform_integer, mu=0, sigma=1).calculate(self.values)
        assert_almost_equal(prior4, truncnorm(-1, 2, loc=0, scale=1).pdf(self.uniform_integer._transform(self.values)))

        prior5 = NormalPrior(self.normal_integer, mu=0, sigma=1).calculate(self.values)
        assert_almost_equal(prior5, norm(loc=0, scale=1).pdf(self.normal_integer._transform(self.values)))

        prior6 = NormalPrior(self.beta_integer, mu=0, sigma=1).calculate(self.values)
        assert_almost_equal(prior6, truncnorm(1, 4, loc=0, scale=1).pdf(self.beta_integer._transform(self.values)))

        prior7 = NormalPrior(self.constant, mu=0, sigma=1).calculate(self.values)
        assert_almost_equal(prior7, np.ones(self.values.shape))

        self.assertRaises(ValueError, NormalPrior(self.categorical, mu=0, sigma=1).calculate, self.values)
        self.assertRaises(ValueError, NormalPrior(self.ordinal, mu=0, sigma=1).calculate, self.values)

    def test_kde_prior(self):
        kde_result = [0.27194, 0.29163, 0.31409, 0.33798, 0.36166, 0.38325, 0.40073, 0.41217, 0.41590, 0.41077]

        prior1 = KdePrior(self.uniform_float, pd.Series([-1, -1, 0, 0.5, 1, 1, 1])).calculate(self.values[::10])
        assert_almost_equal(prior1, kde_result, decimal=5)

        prior2 = KdePrior(self.normal_float, pd.Series([-1, -1, 0, 0.5, 1, 1, 1])).calculate(self.values[::10])
        assert_almost_equal(prior2, kde_result, decimal=5)

        prior3 = KdePrior(self.beta_float, pd.Series([-1, -1, 0, 0.5, 1, 1, 1])).calculate(self.values[::10])
        assert_almost_equal(prior3, kde_result, decimal=5)

        prior4 = KdePrior(self.uniform_integer, pd.Series([-1, -1, 0, 0.5, 1, 1, 1])).calculate(self.values[::10])
        assert_almost_equal(prior4, kde_result, decimal=5)

        prior5 = KdePrior(self.normal_integer, pd.Series([-1, -1, 0, 0.5, 1, 1, 1])).calculate(self.values[::10])
        assert_almost_equal(prior5, kde_result, decimal=5)

        prior6 = KdePrior(self.beta_integer, pd.Series([-1, -1, 0, 0.5, 1, 1, 1])).calculate(self.values[::10])
        assert_almost_equal(prior6, kde_result, decimal=5)

        prior7 = KdePrior(self.constant, pd.Series(['value', 'value', 'value'])).calculate(self.values)
        assert_almost_equal(prior7, np.ones(self.values.shape))

        prior8 = KdePrior(self.categorical, pd.Series(['red', 'red', 'blue'])).calculate(np.array([0, 1, 2, 2]))
        assert_almost_equal(prior8, [2 / 3, 0, 1 / 3, 1 / 3])

        prior9 = KdePrior(self.ordinal, pd.Series(['10', '10', '30'])).calculate(np.array([0, 1, 2, 2]))
        assert_almost_equal(prior9, [2 / 3, 0, 1 / 3, 1 / 3])

    def test_kde_prior_weights(self):
        prior1 = KdePrior(self.uniform_float,
                          pd.Series([-1, -1, 0, 0.5, 1, 1, 1]),
                          weights=pd.Series([0.5, 0.5, 1, 1, 1 / 3, 1 / 3, 1 / 3]),
                          bw=0.1).calculate(np.array([-1, -0.5, 0., 0.5, 1.]))
        assert_almost_equal(prior1, [0.99735, 0, 0.99735, 0.99735, 0.99735], decimal=5)

        prior2 = KdePrior(self.categorical,
                          pd.Series(['red', 'red', 'blue']),
                          weights=pd.Series([0.5, 0.5, 1]),
                          bw=0.1).calculate(np.array([0, 1, 2]))
        assert_almost_equal(prior2, [1, 0, 1], decimal=5)
