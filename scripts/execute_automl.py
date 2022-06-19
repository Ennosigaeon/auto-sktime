import shutil

from matplotlib import pyplot as plt

from autosktime.automl import AutoML
from autosktime.metrics import calculate_loss
from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils.plotting import plot_series

try:
    shutil.rmtree('tmp')
    shutil.rmtree('output')
except FileNotFoundError:
    pass

y = load_airline()
y2 = y.to_timestamp(freq='M')
y3 = y.reset_index(drop=True)

y_train, y_test = temporal_train_test_split(y, test_size=0.2)

automl = AutoML(
    time_left_for_this_task=30,
    per_run_time_limit=10,
    temporary_directory='tmp',
    delete_tmp_folder_after_terminate=False
)

automl.fit(y_train, dataset_name='airline')

y_pred = automl.predict(y_test.index)

print('Ensemble', calculate_loss(y_test, y_pred, automl._task, automl._metric))

fig, ax = plot_series(y_train, y_test, y_pred, labels=['y_train', 'y_test', 'y_pred'])
plt.show()
