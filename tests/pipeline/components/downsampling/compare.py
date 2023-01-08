import unittest

import matplotlib.pyplot as plt
import numpy as np

from autosktime.data.benchmark import FiltrationBenchmark
from autosktime.pipeline.components.downsampling.convolution import ConvolutionDownSampler
from autosktime.pipeline.components.downsampling.elimination import EliminationDownSampler
from autosktime.pipeline.components.downsampling.resampling import ResamplingDownSampling


class AutoMLTest(unittest.TestCase):

    def test_compare_downsampling(self, plot: bool = False):
        X, y = FiltrationBenchmark().get_data()
        X = X.loc[1, 'Differenzdruck']
        y = y.loc[1]

        convolution = ConvolutionDownSampler()
        X_conv, _ = convolution.fit_transform(X, y)
        X_conv_diff = np.abs(X.loc[X_conv.index] - X_conv)
        self.assertAlmostEqual(1.9129752270000022, X_conv_diff.mean())
        self.assertAlmostEqual(2.3768658567672665, X_conv_diff.std())

        eliminiation = EliminationDownSampler()
        X_elim, _ = eliminiation.fit_transform(X, y)
        X_elim_diff = np.abs(X.loc[X_elim.index] - X_elim)
        self.assertAlmostEqual(0., X_elim_diff.mean())
        self.assertAlmostEqual(0., X_elim_diff.std())

        resampling = ResamplingDownSampling()
        X_res, _ = resampling.fit_transform(X, y)
        X_res_diff = np.abs(X.loc[X_res.index] - X_res)
        self.assertAlmostEqual(1.2451703199296607, X_res_diff.mean())
        self.assertAlmostEqual(5.323194406604597, X_res_diff.std())

        if plot:
            fig, ax = plt.subplots(1, 1)
            ax.plot(X, label='Original')
            ax.plot(X_conv, label='Convolution')
            ax.plot(X_elim, label='Elimination')
            ax.plot(X_res, label='Resampling')
            ax.legend()
            plt.show()
