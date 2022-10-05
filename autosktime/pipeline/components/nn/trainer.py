import logging
import math
import numpy as np
import pandas as pd
import tempfile
import time
import torch
from sklearn.utils import check_random_state
from torch.optim import Optimizer
# noinspection PyUnresolvedReferences
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Any, List

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
            use_best_epoch: bool = True,
            random_state: np.random.RandomState = None,
            iterations: int = None,
            config_id: ConfigId = None
    ):
        super().__init__()
        self.patience = patience
        self.tol = tol
        self.use_best_epoch = use_best_epoch

        self.criterion: Optional[torch.nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[_LRScheduler] = None
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
        self.scheduler = data['scheduler']
        self.criterion = torch.nn.MSELoss()

        with tempfile.NamedTemporaryFile() as cache_file:
            self._fit(train_loader=data['train_data_loader'], val_loader=data['val_data_loader'], cache_file=cache_file)

            if self.use_best_epoch:
                self._load_checkpoint(cache_file)

    def _update(self):
        pass

    def _fit(self, train_loader: DataLoader, val_loader: DataLoader, cache_file=None):
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
            val_loss, _ = self._predict(loader=val_loader)

            if self.best_loss_ < val_loss:
                trigger += 1
                abort_msg = f'Performance is decreasing (best {self.best_loss_:1.5f}). Trying {self.patience - trigger} more times'
            elif np.abs(val_loss - self.best_loss_) < self.tol:
                trigger += 1
                abort_msg = f'Performance is not increasing anymore. Trying {self.patience - trigger} more times'
            else:
                trigger = 0
                abort_msg = ''

            if epoch % 10 == 0:
                self.logger.debug(f'Epoch: {self.fitted_epochs_}, train_loss: {train_loss:1.5f}, '
                                  f'val_loss: {val_loss:1.5f}. {abort_msg}')

            if trigger >= self.patience or np.isnan(val_loss):
                self.logger.info(f'Stopping optimization early after {self.fitted_epochs_ + 1} epochs')
                break

            if val_loss < self.best_loss_:
                self._store_checkpoint(cache_file)

            self.fitted_epochs_ += 1
            self.best_loss_ = min(val_loss, self.best_loss_)

    def _train_epoch(self, train_loader: DataLoader) -> float:
        total_loss = 0.0
        self.estimator.train()

        for step, (X, y) in enumerate(train_loader):
            loss, y_hat = self._train_step(X, y)
            total_loss += loss

        self.scheduler.step()
        return total_loss / len(train_loader)

    def _train_step(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, torch.Tensor]:
        # prepare
        X = X.to(self.device)
        y = y.to(self.device)

        # training
        y_hat = self.estimator(X, device=self.device)
        loss = self.criterion(y_hat, y)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return loss.item(), y_hat

    def _store_checkpoint(self, path) -> None:
        if path is not None:
            torch.save({
                'model_state_dict': self.estimator.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, path.name)

    def _load_checkpoint(self, path):
        if path is not None:
            with open(path.name, 'rb') as f:
                checkpoint = torch.load(f)
            self.estimator.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
        patience = UniformIntegerHyperparameter('patience', lower=2, upper=50, default_value=25)
        tol = UniformFloatHyperparameter('tol', 1e-7, 1e-1, default_value=1e-6, log=True)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([patience, tol])
        return cs

    def get_max_iter(self) -> Optional[int]:
        return 128

    def predict(self, data: NN_DATA, y: Any = None, **kwargs) -> np.ndarray:
        loader = data['test_data_loader']
        _, y_hat = self._predict(loader)
        return np.hstack(y_hat).flatten()

    def _predict(self, loader: DataLoader) -> Tuple[float, List[np.ndarray]]:
        self.estimator.eval()
        self.estimator.to(self.device)

        # Batch prediction
        output = []
        total_loss = 0

        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X = X.to(self.device)
                y = y.to(self.device)

                y_hat = self.estimator(X, device=self.device)

                output.append(y_hat.cpu().numpy())
                total_loss += self.criterion(y_hat, y).item()

        avg_loss = total_loss / len(loader)
        return avg_loss, output

    def update(self, data: NN_DATA, y: Any = None, update_params: bool = True):
        return self._fit(train_loader=data['train_data_loader'], val_loader=data['val_data_loader'])
