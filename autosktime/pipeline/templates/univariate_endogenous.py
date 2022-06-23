from itertools import product
from typing import List, Tuple, Dict

from ConfigSpace import ConfigurationSpace, ForbiddenAndConjunction, ForbiddenEqualsClause
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, AutoSktimeChoice
from autosktime.pipeline.components.forecast import ForecasterChoice
from autosktime.pipeline.components.normalizer import NormalizerChoice
from autosktime.pipeline.components.preprocessing.impute import ImputerComponent
from autosktime.pipeline.components.preprocessing.outlier import HampelFilterComponent
from autosktime.pipeline.components.preprocessing.shift import ShiftTransformerComponent
from autosktime.pipeline.templates.base import ConfigurableTransformedTargetForecaster


class UnivariateEndogenousPipeline(ConfigurableTransformedTargetForecaster):

    def get_hyperparameter_search_space(self, dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        if not hasattr(self, 'config_space') or self.config_space is None:
            if dataset_properties is None:
                dataset_properties = self.dataset_properties

            cs = self._get_hyperparameter_search_space(include=self.include, exclude=self.exclude,
                                                       dataset_properties=dataset_properties)

            forecasters = cs.get_hyperparameter('forecaster:__choice__').choices
            normalizers = cs.get_hyperparameter('normalizer:__choice__').choices

            for f, n in product(['ets', 'theta', 'exp_smoothing'], ['scaled_logit']):
                if f not in forecasters or n not in normalizers:
                    continue
                cs.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(cs.get_hyperparameter('forecaster:__choice__'), f),
                    ForbiddenEqualsClause(cs.get_hyperparameter('normalizer:__choice__'), n))
                )

            self.config_space = cs

        return self.config_space

    def _get_pipeline_steps(self, dataset_properties: DatasetProperties) -> List[Tuple[str, AutoSktimeComponent]]:
        steps = []

        default_dataset_properties = {'target_type': 'classification'}
        if dataset_properties is not None and isinstance(dataset_properties, dict):
            default_dataset_properties.update(dataset_properties)

        steps.extend([
            ('shift', ShiftTransformerComponent(random_state=self.random_state)),
            ('outlier', HampelFilterComponent(random_state=self.random_state)),
            ('imputation', ImputerComponent(random_state=self.random_state)),
            ('normalizer', NormalizerChoice(random_state=self.random_state)),
            ('forecaster', ForecasterChoice(random_state=self.random_state))
        ])

        return steps
