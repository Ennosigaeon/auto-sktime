from itertools import product
from typing import List, Tuple, Dict

from ConfigSpace import ConfigurationSpace, ForbiddenAndConjunction, ForbiddenEqualsClause
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, AutoSktimeChoice
from autosktime.pipeline.components.forecast import ForecasterChoice
from autosktime.pipeline.components.normalizer import NormalizerChoice
from autosktime.pipeline.components.preprocessing.impute import ImputerComponent
from autosktime.pipeline.components.preprocessing.outlier import HampelFilterComponent
from autosktime.pipeline.templates.base import ConfigurableTransformedTargetForecaster


class UnivariateEndogenousPipeline(ConfigurableTransformedTargetForecaster):

    def _get_hyperparameter_search_space(
            self,
            include: Dict[str, List[str]] = None,
            exclude: Dict[str, List[str]] = None,
            dataset_properties: DatasetProperties = None
    ):
        """Create the hyperparameter configuration space.

        Parameters
        ----------
        include : dict (optional, default=None)

        Returns
        -------
        cs : ConfigSpace.configuration_space.Configuration
            The configuration space describing the SimpleRegressionClassifier.
        """
        if dataset_properties is None or not isinstance(dataset_properties, DatasetProperties):
            dataset_properties = DatasetProperties()
        self.dataset_properties = dataset_properties

        cs = self._get_base_search_space(include=include, exclude=exclude, dataset_properties=dataset_properties)
        self.configuration_space = cs

        forecasters = cs.get_hyperparameter('forecaster:__choice__').choices
        normalizers = cs.get_hyperparameter('normalizer:__choice__').choices

        for f, n in product(['ets', 'theta', 'exp_smoothing'], ['scaled_logit']):
            if f not in forecasters or n not in normalizers:
                continue
            cs.add_forbidden_clause(ForbiddenAndConjunction(
                ForbiddenEqualsClause(cs.get_hyperparameter('forecaster:__choice__'), f),
                ForbiddenEqualsClause(cs.get_hyperparameter('normalizer:__choice__'), n))
            )

        return cs

    def _get_base_search_space(
            self,
            include: Dict[str, List[str]],
            exclude: Dict[str, List[str]],
            dataset_properties: DatasetProperties,
    ) -> ConfigurationSpace:
        if include is None:
            if self.include is None:
                include = {}
            else:
                include = self.include

        if exclude is None:
            if self.exclude is None:
                exclude = {}
            else:
                exclude = self.exclude

        self._validate_include_exclude_params(include, exclude)

        pipeline = self.steps
        cs = ConfigurationSpace()

        for node_idx, (node_name, node) in enumerate(pipeline):
            # If the node is a choice, we have to figure out which of its choices are actually legal choices
            if isinstance(node, AutoSktimeChoice):
                sub_cs = node.get_hyperparameter_search_space(
                    dataset_properties,
                    include=include.get(node_name), exclude=exclude.get(node_name)
                )
                cs.add_configuration_space(node_name, sub_cs)

            # if the node isn't a choice we can add it immediately
            else:
                cs.add_configuration_space(
                    node_name,
                    node.get_hyperparameter_search_space(dataset_properties),
                )

        return cs

    def _get_pipeline_steps(self, dataset_properties: DatasetProperties) -> List[Tuple[str, AutoSktimeComponent]]:
        steps = []

        default_dataset_properties = {'target_type': 'classification'}
        if dataset_properties is not None and isinstance(dataset_properties, dict):
            default_dataset_properties.update(dataset_properties)

        steps.extend([
            ('outlier', HampelFilterComponent(random_state=self.random_state)),
            ('imputation', ImputerComponent(random_state=self.random_state)),
            ('normalizer', NormalizerChoice(random_state=self.random_state)),
            ('forecaster', ForecasterChoice(random_state=self.random_state))
        ])

        return steps
