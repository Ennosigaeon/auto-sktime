from typing import Any, Dict

import numpy as np
from ConfigSpace.hyperparameters import FloatHyperparameter

from autosktime.metalearning.prior import Prior
from smac.epm.base_epm import AbstractEPM
from smac.optimizer.acquisition import AbstractAcquisitionFunction, PriorAcquisitionFunction as AcqFunc


class PriorAcquisitionFunction(AcqFunc):

    def __init__(
            self,
            model: AbstractEPM,
            acquisition_function: AbstractAcquisitionFunction,
            decay_beta: float,
            priors: Dict[str, Prior],
            **kwargs: Any,
    ):
        super().__init__(
            model=model,
            acquisition_function=acquisition_function,
            decay_beta=decay_beta,
            **kwargs
        )
        self.priors = priors

    def _compute_prior(self, X: np.ndarray) -> np.ndarray:
        prior_values = np.ones((len(X), 1))
        for parameter, X_col in zip(self.hyperparameters.values(), X.T):
            if self.discretize and isinstance(parameter, FloatHyperparameter):
                number_of_bins = int(np.ceil(self.discrete_bins_factor * self.decay_beta / self.iteration_number))
                hp_prior = self._compute_discretized_pdf(parameter, X_col, number_of_bins) + self.prior_floor
            else:
                hp_prior = self.priors[parameter.name].calculate(X_col[:, np.newaxis])

            prior_values *= hp_prior

        return prior_values

    def _compute_discretized_pdf(
            self, parameter: FloatHyperparameter, X_col: np.ndarray, number_of_bins: int
    ) -> np.ndarray:
        prior = self.priors[parameter.name]
        pdf_values = prior.calculate(X_col[:, np.newaxis])

        # Create grid of possible density values
        lower, upper = (0, prior.max_density())
        bin_values = np.linspace(lower, upper, number_of_bins)

        bin_indices = np.clip(
            np.round((pdf_values - lower) * number_of_bins / (upper - lower)), 0, number_of_bins - 1
        ).astype(int)

        prior_values = bin_values[bin_indices]
        return prior_values
