from typing import Tuple, List

from autosktime.pipeline.components.base import AutoSktimeComponent, UpdatablePipeline, SwappedInput
from autosktime.pipeline.components.data_preprocessing import VarianceThresholdComponent
from autosktime.pipeline.components.data_preprocessing.rescaling.standardize import StandardScalerComponent
from autosktime.pipeline.components.data_preprocessing.smooting import SmoothingChoice
from autosktime.pipeline.components.features import FeatureGenerationChoice
from autosktime.pipeline.components.index import AddIndexComponent
from autosktime.pipeline.components.nn.data_loader import SequenceDataLoaderComponent
from autosktime.pipeline.components.nn.lr_scheduler import LearningRateScheduler
from autosktime.pipeline.components.nn.network import NeuralNetworkChoice
from autosktime.pipeline.components.nn.optimizer.optimizer import AdamOptimizer
from autosktime.pipeline.components.nn.trainer import TrainerComponent
from autosktime.pipeline.components.nn.util import DictionaryInput
from autosktime.pipeline.components.normalizer.standardize import TargetStandardizeComponent
from autosktime.pipeline.components.preprocessing.impute import ImputerComponent
from autosktime.pipeline.components.reduction.panel import RecursivePanelReducer
from autosktime.pipeline.templates import PanelRegressionPipeline


class NNPanelRegressionPipeline(PanelRegressionPipeline):

    def _get_pipeline_steps(self) -> List[Tuple[str, AutoSktimeComponent]]:
        pipeline = UpdatablePipeline(steps=[
            ('feature_generation', FeatureGenerationChoice(random_state=self.random_state)),
            ('variance_threshold', VarianceThresholdComponent(random_state=self.random_state)),
            ('scaling', StandardScalerComponent(random_state=self.random_state)),
            ('dict', DictionaryInput()),
            ('data_loader', SequenceDataLoaderComponent(random_state=self.random_state)),
            ('network', NeuralNetworkChoice(random_state=self.random_state)),
            ('optimizer', AdamOptimizer(random_state=self.random_state)),
            ('lr_scheduler', LearningRateScheduler()),
            ('trainer', TrainerComponent(random_state=self.random_state)),
        ])

        steps = [
            # Detrending by collapsing panel data to univariate timeseries by averaging
            ('imputation', ImputerComponent(random_state=self.random_state)),
            ('scaling', TargetStandardizeComponent(random_state=self.random_state)),
            ('reduction', RecursivePanelReducer(
                transformers=[
                    ('smoothing', SwappedInput(SmoothingChoice(random_state=self.random_state))),
                    ('add_index', SwappedInput(AddIndexComponent())),
                ],
                estimator=pipeline,
                random_state=self.random_state,
                dataset_properties=self.dataset_properties)
             ),
        ]
        return steps

    def supports_pynisher(self) -> bool:
        return False
