from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator
from sktime.forecasting.base import ForecastingHorizon


class AutoSktimeComponent(BaseEstimator):
    @staticmethod
    def get_properties(dataset_properties=None):
        """Get the properties of the underlying algorithm.

        Find more information at :ref:`get_properties`

        Parameters
        ----------

        dataset_properties : dict, optional (default=None)

        Returns
        -------
        dict
        """
        raise NotImplementedError()

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        """Return the configuration space of this classification algorithm.

        Parameters
        ----------

        dataset_properties : dict, optional (default=None)

        Returns
        -------
        Configspace.configuration_space.ConfigurationSpace
            The configuration space of this classification algorithm.
        """
        raise NotImplementedError()

    def fit(self, y: pd.Series, X=None, fh: ForecastingHorizon = None):
        """The fit function calls the fit function of the underlying
        sktime model and returns `self`.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters' horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored

        Returns
        -------
        self : returns an instance of self.
        """
        raise NotImplementedError()

    def set_hyperparameters(self, configuration, init_params=None):
        params = configuration.get_dictionary()

        for param, value in params.items():
            if not hasattr(self, param):
                raise ValueError('Cannot set hyperparameter %s for %s because '
                                 'the hyperparameter does not exist.' %
                                 (param, str(self)))
            setattr(self, param, value)

        if init_params is not None:
            for param, value in init_params.items():
                if not hasattr(self, param):
                    raise ValueError('Cannot set init param %s for %s because '
                                     'the init param does not exist.' %
                                     (param, str(self)))
                setattr(self, param, value)

        return self

    def __str__(self):
        name = self.get_properties()['name']
        return "autosktime.pipeline {}".format(name)


class AutoSktimePredictor:

    def predict(
            self,
            fh: Optional[ForecastingHorizon] = None,
            X=None
    ):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(fh=fh, X=X)
