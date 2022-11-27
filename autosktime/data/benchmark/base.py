import abc
import pandas as pd
from typing import Tuple, Dict, List


class Benchmark(abc.ABC):
    folds: int
    start: int

    @abc.abstractmethod
    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass

    @abc.abstractmethod
    def get_train_test_splits(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        pass

    def get_n_splits(self) -> int:
        df, _, _ = self.get_train_test_splits()
        return df.shape[0]

    def score_solutions(self, y_pred, y_test):
        from autosktime.metrics import STRING_TO_METRIC

        if not hasattr(self, 'performance'):
            self.performance: Dict[str, List[float]] = {
                'rmse': [],
                'wrmse': [],
                'mae': [],
                'wmae': [],
                'me': [],
                'std': [],
                'maare': [],
                'relph': [],
                'phrate': [],
                'cra': [],
            }

        for metric_name in self.performance.keys():
            metric = STRING_TO_METRIC[metric_name](start=self.start)
            self.performance[metric_name].append(metric(y_test, y_pred))
