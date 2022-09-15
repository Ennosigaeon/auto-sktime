import logging
import math
import time
from typing import Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from sklearn.utils import check_random_state
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, UniformFloatHyperparameter
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, IGNORES_EXOGENOUS_X, \
    SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimeRegressionAlgorithm
from autosktime.pipeline.components.nn.util import NN_DATA
from autosktime.pipeline.util import Int64Index
from autosktime.util.backend import ConfigContext, ConfigId


class TrainerComponent(AutoSktimeRegressionAlgorithm):
    estimator: torch.nn.Module = None

    def __init__(
            self,
            patience: int = 5,
            tol: float = 1e-6,
            random_state: np.random.RandomState = None,
            iterations: int = None,
            config_id: ConfigId = None
    ):
        super().__init__()
        self.patience = patience
        self.tol = tol

        self.criterion: Optional[torch.nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.device: Optional[torch.device] = None

        self.random_state = random_state if random_state is not None else check_random_state(1)
        self.iterations = iterations
        self.config_id = config_id

        self.logger = logging.getLogger('NeuralNetwork')
        self.fitted_epochs_ = 0
        self.best_loss_ = np.inf

    def fit(self, data: NN_DATA, y: Any = None, **kwargs):
        self.device = data['device']
        self.estimator = data['network'].to(self.device)
        self.optimizer = data['optimizer']
        self.criterion = torch.nn.MSELoss()

        return self._fit(train_loader=data['train_data_loader'], val_loader=data['val_data_loader'])

    def _update(self):
        pass

    def _fit(self, train_loader: DataLoader, val_loader: DataLoader):
        config: ConfigContext = ConfigContext.instance()
        iterations = self.iterations or config.get_config(self.config_id, 'iterations') or self.get_max_iter()
        cutoff = config.get_config(self.config_id, 'cutoff') or math.inf
        start = config.get_config(self.config_id, 'start') or 0

        trigger = 0

        for epoch in range(iterations - self.fitted_epochs_):
            if time.time() - start > cutoff:
                self.logger.info(f'Aborting fitting after {self.fitted_epochs_} epochs due to timeout')
                break

            train_loss = self._train_epoch(train_loader=train_loader)
            val_loss = self._test_model(data_loader=val_loader)
            self.logger.info(f'Epoch: {self.fitted_epochs_}, train_loss: {train_loss:1.5f}, '
                             f'val_loss: {val_loss:1.5f}')

            if self.best_loss_ < val_loss:
                trigger += 1
                self.logger.debug(f'Performance is decreasing. Trying {self.patience - trigger} more times')
            elif np.abs(val_loss - self.best_loss_) < self.tol:
                trigger += 1
                self.logger.info(f'Performance is not increasing anymore. Trying {self.patience - trigger} more times')
            else:
                trigger = 0

            if trigger >= self.patience:
                self.logger.info(f'Stopping optimization early after {self.fitted_epochs_ + 1} epochs')
                break

            self.fitted_epochs_ += 1
            self.best_loss_ = min(val_loss, self.best_loss_)

    def _train_epoch(self, train_loader: DataLoader) -> float:
        total_loss = 0.0
        self.estimator.train()

        for step, (data, targets) in enumerate(train_loader):
            loss, outputs = self._train_step(data, targets)
            total_loss += loss
        return total_loss / len(train_loader)

    def _train_step(self, data: torch.Tensor, targets: torch.Tensor) -> Tuple[float, torch.Tensor]:
        # prepare
        data = data.to(self.device)
        targets = targets.to(self.device)

        # training
        outputs = self.estimator(data, device=self.device)
        loss = self.criterion(outputs, targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # TODO remove once pytorch > 1.12.0 has been released (see https://github.com/pytorch/pytorch/pull/80345)
        if not hasattr(self.optimizer, '_warned_capturable_if_run_uncaptured'):
            self.optimizer._warned_capturable_if_run_uncaptured = True

        self.optimizer.step()

        return loss.item(), outputs

    def _test_model(self, data_loader: DataLoader) -> float:
        total_loss = 0

        self.estimator.eval()
        with torch.no_grad():
            for X, y in data_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                output = self.estimator(X, device=self.device)
                total_loss += self.criterion(output, y).item()

        avg_loss = total_loss / len(data_loader)
        return avg_loss

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        patience = UniformIntegerHyperparameter('patience', lower=1, upper=5, default_value=3)
        tol = UniformFloatHyperparameter('tol', 1e-5, 1e-1, default_value=1e-4, log=True)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([patience, tol])
        return cs

    def get_max_iter(self) -> Optional[int]:
        return 16

    def predict(self, data: NN_DATA, y: Any = None, **kwargs) -> torch.Tensor:
        loader = data['test_data_loader']
        self.estimator.eval()
        self.estimator.to(self.device)

        # Batch prediction
        output = torch.tensor([])

        with torch.no_grad():
            for X, _ in loader:
                X = X.to(self.device)

                y_star = self.estimator(X, device=self.device).cpu()
                output = torch.cat((output, y_star), 0)

        return output.numpy()

    def update(self, data: NN_DATA, y: Any = None, update_params: bool = True):
        return self._fit(train_loader=data['train_data_loader'], val_loader=data['val_data_loader'])
