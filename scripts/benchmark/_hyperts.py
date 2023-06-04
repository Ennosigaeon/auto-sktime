import pandas as pd
from hyperts import make_experiment
from hyperts.framework.compete import TSPipeline

from scripts.benchmark.util import generate_fh


def evaluate_hyperts(y: pd.Series, fh: int):
    fh = pd.DataFrame(generate_fh(y.index, fh).to_timestamp(), columns=['index'])
    y = y.reset_index()

    model: TSPipeline = make_experiment(y, task='univariate-forecast', timestamp='index').run()

    y_pred = model.predict(fh).set_index('index')
    return y_pred['y'], None
