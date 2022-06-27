import shutil
import unittest
from typing import Dict
from unittest import mock

from autosktime.constants import UNIVARIATE_FORECAST
from smac.runhistory.runhistory import RunInfo
from smac.stats.stats import Stats

from ConfigSpace import Configuration
from autosktime.automl_common.common.utils.backend import create
from autosktime.data import DataManager
from autosktime.data.splitter import HoldoutSplitter
from autosktime.evaluation import ExecuteTaFunc
from autosktime.metrics import MeanAbsolutePercentageError
from autosktime.pipeline.templates.univariate_endogenous import UnivariateEndogenousPipeline
from sktime.datasets import load_airline


class TestExecuteTAFunc(unittest.TestCase):

    def _test_configuration(self, config_dict: Dict):
        y = load_airline()
        datamanager = DataManager(UNIVARIATE_FORECAST, y, None, 'airlines')

        try:
            backend = create('test_tmp', 'test_output', 'prefix')

            backend.save_datamanager(datamanager)
            metric = MeanAbsolutePercentageError()
            splitter = HoldoutSplitter()

            cs = UnivariateEndogenousPipeline(dataset_properties=datamanager.dataset_properties).config_space
            config = Configuration(cs, config_dict)
            config.config_id = 0

            scenario_mock = mock.Mock()
            scenario_mock.wallclock_limit = 300
            stats = Stats(scenario_mock)

            ta_executor = ExecuteTaFunc(backend, 1, splitter, metric, stats)
            info, value = ta_executor.run_wrapper(
                RunInfo(config=config, instance=None, instance_specific='', seed=1, cutoff=300, capped=True)
            )
            print(value)
        finally:
            try:
                shutil.rmtree('test_tmp')
                shutil.rmtree('test_output')
            except FileNotFoundError:
                pass
