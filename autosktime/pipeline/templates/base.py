from abc import ABC
from typing import Dict, Any, List, Tuple

from sktime.forecasting.compose import TransformedTargetForecaster

from ConfigSpace import Configuration, ConfigurationSpace
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, AutoSktimeChoice


class ConfigurablePipeline(ABC):

    def _init(
            self,
            config: Configuration = None,
            dataset_properties: DatasetProperties = None,
            include: Dict[str, List[str]] = None,
            exclude: Dict[str, List[str]] = None,
            random_state=None,
            init_params: Dict[str, Any] = None
    ):
        self.init_params = init_params if init_params is not None else {}
        self.include = include if include is not None else {}
        self.exclude = exclude if exclude is not None else {}
        self.dataset_properties = dataset_properties
        self.random_state = random_state

        self.steps = self._get_pipeline_steps(dataset_properties=dataset_properties)

        self._validate_include_exclude_params(self.include, self.exclude)

        self.config_space = self.get_hyperparameter_search_space(self.dataset_properties)

        if config is None:
            self.config = self.config_space.get_default_configuration()
        else:
            if isinstance(config, dict):
                config = Configuration(self.config_space, config)
            if self.config_space != config.configuration_space:
                import difflib
                diff = difflib.unified_diff(
                    str(self.config_space).splitlines(),
                    str(config.configuration_space).splitlines()
                )
                raise ValueError('Configuration passed does not come from the same configuration space. Differences '
                                 'are: {}'.format('\n'.join(diff)))
            self.config = config

        self.set_hyperparameters(self.config, init_params=init_params)
        self._additional_run_info = {}

    def _validate_include_exclude_params(self, include: Dict[str, List[str]], exclude: Dict[str, List[str]]) -> None:
        if include is not None and exclude is not None:
            for key in include.keys():
                if key in exclude.keys():
                    raise ValueError(f"Cannot specify include and exclude for same step '{key}'.")

        supported_steps: Dict[str, AutoSktimeChoice] = {
            name: choice for name, choice in self.steps
            if isinstance(choice, AutoSktimeChoice)
        }

        for arg, argument in [('include', include), ('exclude', exclude)]:
            if not argument:
                continue
            for key in argument.keys():
                if key not in supported_steps:
                    raise ValueError(
                        f"The provided key '{key}' in the '{arg}' argument is not valid. "
                        f"The only supported keys for this task are {list(supported_steps.keys())}")

                candidate_components = argument[key]
                if not (isinstance(candidate_components, list) and candidate_components):
                    raise ValueError(
                        f"The provided value of the key '{key}' in the '{arg}' argument is not valid. "
                        f"The value must be a non-empty list.")

                available_components = list(
                    supported_steps[key].get_available_components(dataset_properties=self.dataset_properties).keys()
                )
                for component in candidate_components:
                    if component not in available_components:
                        raise ValueError(
                            f"The provided component '{component}' for the key '{key}' in the '{arg}' argument is not "
                            f"valid. The supported components for the step '{key}' for this task are "
                            f"{available_components}")

    def set_hyperparameters(
            self,
            configuration: Configuration,
            init_params: Dict[str, Any] = None
    ):
        self.config = configuration

        for node_idx, (node_name, node) in enumerate(self.steps):
            sub_configuration_space = node.get_hyperparameter_search_space(dataset_properties=self.dataset_properties)
            sub_config_dict = {}
            for param in configuration:
                if param.startswith(f'{node_name}:'):
                    value = configuration[param]
                    new_name = param.replace(f'{node_name}:', '', 1)
                    sub_config_dict[new_name] = value

            sub_configuration = Configuration(sub_configuration_space, values=sub_config_dict)

            if init_params is not None:
                sub_init_params_dict = {}
                for param in init_params:
                    if param.startswith(f'{node_name}:'):
                        value = init_params[param]
                        new_name = param.replace(f'{node_name}:', '', 1)
                        sub_init_params_dict[new_name] = value
            else:
                sub_init_params_dict = None

            if isinstance(node, (AutoSktimeComponent, ConfigurableTransformedTargetForecaster)):
                node.set_hyperparameters(configuration=sub_configuration, init_params=sub_init_params_dict)
            else:
                raise NotImplementedError('Not supported yet!')

        return self

    def get_hyperparameter_search_space(self, dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        """Return the configuration space for the CASH problem.

        Returns
        -------
        cs : ConfigSpace.configuration_space.Configuration
        """
        if not hasattr(self, 'config_space') or self.config_space is None:
            self.config_space = self._get_hyperparameter_search_space(
                include=self.include, exclude=self.exclude,
                dataset_properties=dataset_properties)
        return self.config_space

    def _get_hyperparameter_search_space(
            self,
            include: Dict[str, List[str]],
            exclude: Dict[str, List[str]],
            dataset_properties: DatasetProperties = None,
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
        raise NotImplementedError()


class ConfigurableTransformedTargetForecaster(TransformedTargetForecaster, ConfigurablePipeline, ABC):

    def __init__(
            self,
            config: Configuration = None,
            steps: List[Tuple[str, AutoSktimeComponent]] = None,
            dataset_properties: DatasetProperties = None,
            include: Dict[str, List[str]] = None,
            exclude: Dict[str, List[str]] = None,
            random_state=None,
            init_params: Dict[str, Any] = None
    ):
        self._init(config, steps, dataset_properties, include, exclude, random_state, init_params)
        super().__init__(self.steps)

    def _transform(self, X, y=None):
        # Only implemented for type checker, not actually used
        return super(ConfigurableTransformedTargetForecaster, self)._transform(X, y)

    def _inverse_transform(self, X, y=None):
        # Only implemented for type checker, not actually used
        return super(ConfigurableTransformedTargetForecaster, self)._inverse_transform(X, y)

    def get_fitted_params(self):
        # Only implemented for type checker, not actually used
        return super(ConfigurableTransformedTargetForecaster, self).get_fitted_params()


BasePipeline = ConfigurableTransformedTargetForecaster
