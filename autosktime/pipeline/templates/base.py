import abc
from abc import ABC
from typing import Dict, Any, List, Tuple

from ConfigSpace import Configuration, ConfigurationSpace
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, AutoSktimeChoice, AutoSktimePredictor
from sktime.forecasting.compose import TransformedTargetForecaster


class ConfigurableTransformedTargetForecaster(TransformedTargetForecaster, ABC):

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
        self.init_params = init_params if init_params is not None else {}
        self.include = include if include is not None else {}
        self.exclude = exclude if exclude is not None else {}
        self.dataset_properties = dataset_properties if dataset_properties is not None else DatasetProperties()
        self.random_state = random_state

        if steps is None:
            self.steps = self._get_pipeline_steps(dataset_properties=dataset_properties)
        else:
            self.steps = steps

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

        super().__init__(steps=self.steps)

        self._additional_run_info = {}

    def _validate_include_exclude_params(self, include: Dict[str, List[str]], exclude: Dict[str, List[str]]) -> None:
        if include is not None and exclude is not None:
            for key in include.keys():
                if key in exclude.keys():
                    raise ValueError("Cannot specify include and exclude for same step '{}'.".format(key))

        supported_steps: Dict[str, AutoSktimeChoice] = {
            name: choice for name, choice in self.steps
            if isinstance(choice, AutoSktimeChoice)
        }

        for arg, argument in [('include', include), ('exclude', exclude)]:
            if not argument:
                continue
            for key in argument.keys():
                if key not in supported_steps:
                    raise ValueError("The provided key '{}' in the '{}' argument is not valid. The only supported keys "
                                     "for this task are {}".format(key, arg, list(supported_steps.keys())))

                candidate_components = argument[key]
                if not (isinstance(candidate_components, list) and candidate_components):
                    raise ValueError("The provided value of the key '{}' in the '{}' argument is not valid. The value "
                                     "must be a non-empty list.".format(key, arg))

                available_components = list(
                    supported_steps[key].get_available_components(dataset_properties=self.dataset_properties).keys()
                )
                for component in candidate_components:
                    if component not in available_components:
                        raise ValueError("The provided component '{}' for the key '{}' in the '{}' argument is not "
                                         "valid. The supported components for the step '{}' for this task are {}"
                                         .format(component, key, arg, key, available_components))

    def set_hyperparameters(
            self,
            configuration: Configuration,
            init_params: Dict[str, Any] = None
    ) -> 'ConfigurableTransformedTargetForecaster':
        self.config = configuration

        for node_idx, (node_name, node) in enumerate(self.steps):
            sub_configuration_space = node.get_hyperparameter_search_space(dataset_properties=self.dataset_properties)
            sub_config_dict = {}
            for param in configuration:
                if param.startswith('{}:'.format(node_name)):
                    value = configuration[param]
                    new_name = param.replace('{}:'.format(node_name), '', 1)
                    sub_config_dict[new_name] = value

            sub_configuration = Configuration(sub_configuration_space, values=sub_config_dict)

            if init_params is not None:
                sub_init_params_dict = {}
                for param in init_params:
                    if param.startswith('{}:'.format(node_name)):
                        value = init_params[param]
                        new_name = param.replace('{}:'.format(node_name), '', 1)
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

    @abc.abstractmethod
    def _get_hyperparameter_search_space(
            self,
            include: Dict[str, List[str]] = None,
            exclude: Dict[str, List[str]] = None,
            dataset_properties: DatasetProperties = None
    ) -> ConfigurationSpace:
        raise NotImplementedError()

    def _get_pipeline_steps(self, dataset_properties: DatasetProperties) -> List[Tuple[str, AutoSktimeComponent]]:
        raise NotImplementedError()

    def _transform(self, X, y=None):
        # Only implemented for type checker, not actually used
        return super(ConfigurableTransformedTargetForecaster, self)._transform(X, y)

    def _inverse_transform(self, X, y=None):
        # Only implemented for type checker, not actually used
        return super(ConfigurableTransformedTargetForecaster, self)._inverse_transform(X, y)

    def get_fitted_params(self):
        # Only implemented for type checker, not actually used
        return super(ConfigurableTransformedTargetForecaster, self).get_fitted_params()