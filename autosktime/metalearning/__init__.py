from datetime import time

from autosktime.metrics import BaseMetric


def suggest_via_metalearning(
        meta_base, dataset_name: str, metric: BaseMetric, task, sparse,
        num_initial_configurations, logger):
    if task == MULTILABEL_CLASSIFICATION:
        task = MULTICLASS_CLASSIFICATION

    task = TASK_TYPES_TO_STRING[task]

    logger.info(task)

    start = time.time()
    ml = MetaLearningOptimizer(
        dataset_name=dataset_name,
        configuration_space=meta_base.configuration_space,
        meta_base=meta_base,
        distance='l1',
        seed=1,
        logger=logger, )
    logger.info('Reading meta-data took %5.2f seconds',
                time.time() - start)
    runs = ml.metalearning_suggest_all(exclude_double_configurations=True)
    return runs[:num_initial_configurations]
