import numpy as np


def mean(x: np.ndarray) -> np.ndarray:
    """
    Returns the mean of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return x.mean(axis=1)


def variance(x: np.ndarray) -> np.ndarray:
    """
    Returns the variance of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.var(x, axis=1)


def standard_deviation(x: np.ndarray) -> np.ndarray:
    """
    Returns the standard deviation of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.std(x, axis=1)


def length(x: np.ndarray) -> np.ndarray:
    """
    Returns the length of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: int
    """
    return np.ones((x.shape[0], x.shape[2])) * x.shape[1]


def skewness(x: np.ndarray) -> np.ndarray:
    """
    Returns the sample skewness of x (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G1).

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    from scipy.stats import skew
    return skew(x, axis=1)


def median(x: np.ndarray) -> np.ndarray:
    """
    Returns the median of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.median(x, axis=1)


def kurtosis(x: np.ndarray) -> np.ndarray:
    """
    Returns the kurtosis of x (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G2).

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    from scipy.stats import kurtosis as k_
    return k_(x, axis=1)


def minimum(x: np.ndarray) -> np.ndarray:
    """
    Calculates the lowest value of the time series x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.min(x, axis=1)


def maximum(x: np.ndarray) -> np.ndarray:
    """
    Calculates the highest value of the time series x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.max(x, axis=1)


def last_location_of_maximum(x: np.ndarray) -> np.ndarray:
    """
    Returns the relative last location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return 1.0 - np.argmax(x, axis=1) / x.shape[1]


def mean_second_derivative_central(x: np.ndarray) -> np.ndarray:
    """
    Returns the mean value of a central approximation of the second derivative

    .. math::

        \\frac{1}{2(n-2)} \\sum_{i=1,\\ldots, n-1}  \\frac{1}{2} (x_{i+2} - 2 \\cdot x_{i+1} + x_i)

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return (x[:, -1] - x[:, -2] - x[:, 1] + x[:, 0]) / (2 * (x.shape[1] - 2)) if x.shape[1] > 2 else 0


def abs_energy(x: np.ndarray) -> np.ndarray:
    """
    Returns the absolute energy of the time series which is the sum over the squared values

    .. math::

        E = \\sum_{i=1,\\ldots, n} x_i^2

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.power(x, 2).sum(axis=1)


def last_location_of_maximum(x: np.ndarray) -> np.ndarray:
    """
    Returns the relative last location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return 1.0 - np.argmax(x[::-1], axis=1) / x.shape[1] if x.shape[1] > 0 else 0


def first_location_of_maximum(x: np.ndarray) -> np.ndarray:
    """
    Returns the first location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.argmax(x, axis=1) / x.shape[1] if x.shape[1] > 0 else 0


def first_location_of_minimum(x: np.ndarray) -> np.ndarray:
    """
    Returns the first location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.argmin(x, axis=1) / x.shape[1] if x.shape[1] > 0 else 0


def last_location_of_minimum(x: np.ndarray) -> np.ndarray:
    """
    Returns the last location of the minimal value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return 1.0 - np.argmin(x[::-1], axis=1) / x.shape[1] if x.shape[1] > 0 else 0


def mean_abs_change(x: np.ndarray) -> np.ndarray:
    """
    Average over first differences.

    Returns the mean over the absolute differences between subsequent time series values which is

    .. math::

        \\frac{1}{n-1} \\sum_{i=1,\\ldots, n-1} | x_{i+1} - x_{i}|


    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.mean(np.abs(np.diff(x, axis=1)), axis=1)


def mean_change(x: np.ndarray) -> np.ndarray:
    """
    Average over time series differences.

    Returns the mean over the differences between subsequent time series values which is

    .. math::

        \\frac{1}{n-1} \\sum_{i=1,\\ldots, n-1}  x_{i+1} - x_{i} = \\frac{1}{n-1} (x_{n} - x_{1})

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return (x[:, -1, :] - x[:, 0, :]) / (x.shape[1] - 1) if x.shape[1] > 1 else 0


def count_above_mean(x: np.ndarray) -> np.ndarray:
    """
    Returns the number of values in x that are higher than the mean of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return count_below_mean(x, invert=True)


def count_below_mean(x: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    Returns the number of values in x that are lower than the mean of x

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    m = np.mean(x, axis=1)
    res = np.empty((x.shape[0], x.shape[2]))

    for i in range(x.shape[2]):
        m_ = np.repeat(np.atleast_2d(m[:, i]), x.shape[1], axis=0).T
        below = x[:, :, i] > m_ if invert else x[:, :, i] < m_
        res[:, i] = np.sum(below, axis=1)

    return res


def count_below(x: np.ndarray, t: float, invert: bool = False) -> np.ndarray:
    """
    Returns the percentage of values in x that are lower than t

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param t: value used as threshold
    :type t: float

    :return: the value of this feature
    :return type: float
    """
    res = np.empty((x.shape[0], x.shape[2]))

    for i in range(x.shape[2]):
        below = x[:, :, i] > t if invert else x[:, :, i] < t
        res[:, i] = np.sum(below, axis=1) / x.shape[1]

    return res


def count_above(x: np.ndarray, t: float) -> np.ndarray:
    """
    Returns the percentage of values in x that are higher than t

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param t: value used as threshold
    :type t: float

    :return: the value of this feature
    :return type: float
    """
    return count_below(x, t, invert=True)


def range_count(x: np.ndarray, min: float, max: float) -> np.ndarray:
    """
    Count observed values within the interval [min, max).

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param min: the inclusive lower bound of the range
    :type min: int or float
    :param max: the exclusive upper bound of the range
    :type max: int or float
    :return: the count of values within the range
    :rtype: int
    """
    res = np.empty((x.shape[0], x.shape[2]))

    for i in range(x.shape[2]):
        between = (min <= x[:, :, i]) & (x[:, :, i] < max)
        res[:, i] = np.sum(between, axis=1) / x.shape[1]

    return res


def sum_values(x: np.ndarray) -> np.ndarray:
    """
    Calculates the sum over the time series values

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.sum(x, axis=1) if x.shape[1] > 0 else 0


def absolute_sum_of_changes(x: np.ndarray) -> np.ndarray:
    """
    Returns the sum over the absolute value of consecutive changes in the series x

    .. math::

        \\sum_{i=1, \\ldots, n-1} \\mid x_{i+1}- x_i \\mid

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    return np.sum(np.abs(np.diff(x, axis=1)), axis=1)


def variance_larger_than_standard_deviation(x: np.ndarray) -> np.ndarray:
    """
    Is variance higher than the standard deviation?

    Boolean variable denoting if the variance of x is greater than its standard deviation. Is equal to variance of x
    being larger than 1

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: bool
    """
    y = np.var(x, axis=1)
    return y > np.sqrt(y)


def percentage_of_reoccurring_values_to_all_values(x: np.ndarray) -> np.ndarray:
    """
    Returns the percentage of values that are present in the time series
    more than once.

        len(different values occurring more than once) / len(different values)

    This means the percentage is normalized to the number of unique values,
    in contrast to the percentage_of_reoccurring_datapoints_to_all_datapoints.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    counts = _count_values(x)
    return 1 - counts / float(x.shape[1])


def has_duplicate(x: np.ndarray) -> np.ndarray:
    """
    Checks if any value in x occurs more than once

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: bool
    """
    counts = _count_values(x)
    return counts != x.shape[1]


def ratio_beyond_r_sigma(x: np.ndarray, r: float) -> np.ndarray:
    """
    Ratio of values that are more than r * std(x) (so r times sigma) away from the mean of x.

    :param x: the time series to calculate the feature of
    :type x: iterable
    :param r: the ratio to compare with
    :type r: float
    :return: the value of this feature
    :return type: float
    """
    mean = np.repeat(np.mean(x, axis=1)[:, np.newaxis, :], x.shape[1], axis=1)
    std = np.repeat(np.std(x, axis=1)[:, np.newaxis, :], x.shape[1], axis=1)

    return np.sum(np.abs(x - mean) > r * std, axis=1) / x.shape[1]


def large_standard_deviation(x: np.ndarray, r: float) -> np.ndarray:
    """
    Does time series have *large* standard deviation?

    Boolean variable denoting if the standard dev of x is higher than 'r' times the range = difference between max and
    min of x. Hence it checks if

    .. math::

        std(x) > r * (max(X)-min(X))

    According to a rule of the thumb, the standard deviation should be a forth of the range of the values.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param r: the percentage of the range to compare with
    :type r: float
    :return: the value of this feature
    :return type: bool
    """
    return np.std(x, axis=1) > (r * (np.max(x, axis=1) - np.min(x, axis=1)))


def quantile(x: np.ndarray, q: float) -> np.ndarray:
    """
    Calculates the q quantile of x. This is the value of x greater than q% of the ordered values from x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param q: the quantile to calculate
    :type q: float
    :return: the value of this feature
    :return type: float
    """
    return np.quantile(x, q, axis=1) if x.shape[1] > 0 else 0


def number_crossing_m(x: np.ndarray, m: float) -> np.ndarray:
    """
    Calculates the number of crossings of x on m. A crossing is defined as two sequential values where the first value
    is lower than m and the next is greater, or vice-versa. If you set m to zero, you will get the number of zero
    crossings.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param m: the threshold for the crossing
    :type m: float
    :return: the value of this feature
    :return type: int
    """
    positive = x > m
    return np.sum(np.diff(positive, axis=1), axis=1)


def cid_ce(x: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This function calculator is an estimate for a time series complexity [1] (A more complex time series has more peaks,
    valleys etc.). It calculates the value of

    .. math::

        \\sqrt{ \\sum_{i=1}^{n-1} ( x_{i} - x_{i-1})^2 }

    .. rubric:: References

    |  [1] Batista, Gustavo EAPA, et al (2014).
    |  CID: an efficient complexity-invariant distance for time series.
    |  Data Mining and Knowledge Discovery 28.3 (2014): 634-669.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param normalize: should the time series be z-transformed?
    :type normalize: bool

    :return: the value of this feature
    :return type: float
    """
    if normalize:
        s = np.repeat(np.std(x, axis=1)[:, np.newaxis, :], x.shape[1], axis=1)
        if np.min(s) != 0:
            m = np.repeat(np.mean(x, axis=1)[:, np.newaxis, :], x.shape[1], axis=1)
            x = (x - m) / s
        else:
            return np.zeros((x.shape[0], x.shape[2]))

    x = np.diff(x, axis=1)
    # dot = np.apply_along_axis(lambda x: np.dot(x, x), 1, x)
    dot = np.power(x, 2).sum(axis=1)
    return np.sqrt(dot)


def latest(x: np.ndarray) -> np.ndarray:
    """
    This function returns the latest value of each feature.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray

    :return: the value of this feature
    :return type: float
    """
    return x[:, -1, :]


def crest(x: np.ndarray, pctarr: float = 100.0, intep: str = 'midpoint') -> np.ndarray:
    peak = np.percentile(np.abs(x), pctarr, method=intep, axis=1)
    sig = x.std(axis=1)

    CF = np.divide(peak, sig, out=np.zeros_like(peak), where=sig != 0)
    return CF


def c3(x: np.ndarray, lag: int) -> np.ndarray:
    """
    Uses c3 statistics to measure non linearity in the time series

    This function calculates the value of

    .. math::

        \\frac{1}{n-2lag} \\sum_{i=1}^{n-2lag} x_{i + 2 \\cdot lag} \\cdot x_{i + lag} \\cdot x_{i}

    which is

    .. math::

        \\mathbb{E}[L^2(X) \\cdot L(X) \\cdot X]

    where :math:`\\mathbb{E}` is the mean and :math:`L` is the lag operator. It was proposed in [1] as a measure of
    non linearity in the time series.

    .. rubric:: References

    |  [1] Schreiber, T. and Schmitz, A. (1997).
    |  Discrimination power of measures for nonlinearity in a time series
    |  PHYSICAL REVIEW E, VOLUME 55, NUMBER 5

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param lag: the lag that should be used in the calculation of the feature
    :type lag: int
    :return: the value of this feature
    :return type: float
    """
    n = x.shape[1]
    if 2 * lag >= n:
        return np.zeros((x.shape[0], x.shape[2]))
    else:
        return np.mean((np.roll(x, 2 * -lag, axis=1) * np.roll(x, -lag, axis=1) * x)[:, 0: (n - 2 * lag), :], axis=1)


def symmetry_looking(x: np.ndarray, r: int) -> np.ndarray:
    """
    Boolean variable denoting if the distribution of x *looks symmetric*. This is the case if

    .. math::

        | mean(X)-median(X)| < r * (max(X)-min(X))

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"r": x} with x (float) is the percentage of the range to compare with
    :type param: list
    :return: the value of this feature
    :return type: bool
    """
    mean_median_difference = np.abs(np.mean(x, axis=1) - np.median(x, axis=1))
    max_min_difference = np.max(x, axis=1) - np.min(x, axis=1)
    return mean_median_difference < r * max_min_difference


def time_reversal_asymmetry_statistic(x: np.ndarray, lag: int) -> np.ndarray:
    """
    Returns the time reversal asymmetry statistic.

    This function calculates the value of

    .. math::

        \\frac{1}{n-2lag} \\sum_{i=1}^{n-2lag} x_{i + 2 \\cdot lag}^2 \\cdot x_{i + lag} - x_{i + lag} \\cdot  x_{i}^2

    which is

    .. math::

        \\mathbb{E}[L^2(X)^2 \\cdot L(X) - L(X) \\cdot X^2]

    where :math:`\\mathbb{E}` is the mean and :math:`L` is the lag operator. It was proposed in [1] as a
    promising feature to extract from time series.

    .. rubric:: References

    |  [1] Fulcher, B.D., Jones, N.S. (2014).
    |  Highly comparative feature-based time-series classification.
    |  Knowledge and Data Engineering, IEEE Transactions on 26, 3026–3037.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param lag: the lag that should be used in the calculation of the feature
    :type lag: int
    :return: the value of this feature
    :return type: float
    """
    n = x.shape[1]
    if 2 * lag >= n:
        return np.zeros((x.shape[0], x.shape[1]))
    else:
        one_lag = np.roll(x, -lag, axis=1)
        two_lag = np.roll(x, 2 * -lag, axis=1)
        return np.mean(
            (two_lag * two_lag * one_lag - one_lag * x * x)[:, 0:(n - 2 * lag), :], axis=1
        )


def fft_coefficient(x: np.ndarray, coeff: int, attr: str) -> np.ndarray:
    """
    Calculates the fourier coefficients of the one-dimensional discrete Fourier Transform for real input by fast
    fourier transformation algorithm

    .. math::
        A_k =  \\sum_{m=0}^{n-1} a_m \\exp \\left \\{ -2 \\pi i \\frac{m k}{n} \\right \\}, \\qquad k = 0,
        \\ldots , n-1.

    The resulting coefficients will be complex, this feature calculator can return the real part (attr=="real"),
    the imaginary part (attr=="imag), the absolute value (attr=""abs) and the angle in degrees (attr=="angle).

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"coeff": x, "attr": s} with x int and x >= 0, s str and in ["real", "imag",
        "abs", "angle"]
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """
    fft = np.fft.rfft(x, axis=1)

    def complex_agg(x: np.ndarray, agg: str) -> np.ndarray:
        if agg == "real":
            return x.real
        elif agg == "imag":
            return x.imag
        elif agg == "abs":
            return np.abs(x)
        elif agg == "angle":
            return np.angle(x, deg=True)

    return complex_agg(fft[:, coeff, :], attr)


def autocorrelation(x: np.ndarray, lag: int) -> np.ndarray:
    """
    Calculates the autocorrelation of the specified lag, according to the formula [1]

    .. math::

        \\frac{1}{(n-l)\\sigma^{2}} \\sum_{t=1}^{n-l}(X_{t}-\\mu )(X_{t+l}-\\mu)

    where :math:`n` is the length of the time series :math:`X_i`, :math:`\\sigma^2` its variance and :math:`\\mu` its
    mean. `l` denotes the lag.

    .. rubric:: References

    [1] https://en.wikipedia.org/wiki/Autocorrelation#Estimation

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param lag: the lag
    :type lag: int
    :return: the value of this feature
    :return type: float
    """
    # This is important: If a series is passed, the product below is calculated
    # based on the index, which corresponds to squaring the series.
    if x.shape[1] < lag:
        return np.zeros((x.shape[0], x.shape[1]))
    # Slice the relevant subseries based on the lag
    y1 = x[:, :(x.shape[1] - lag), :]
    y2 = x[:, lag:, :]
    # Subtract the mean of the whole series x
    x_mean = np.repeat(np.mean(x, axis=1)[:, np.newaxis, :], x.shape[1] - lag, axis=1)

    # The result is sometimes referred to as "covariation"
    sum_product = np.sum((y1 - x_mean) * (y2 - x_mean), axis=1)
    # Return the normalized unbiased covariance
    v = np.var(x, axis=1)

    return np.nan_to_num(sum_product / ((x.shape[1] - lag) * v), 0)


def number_peaks(x: np.ndarray, n: int) -> np.ndarray:
    """
    Calculates the number of peaks of at least support n in the time series x. A peak of support n is defined as a
    subsequence of x where a value occurs, which is bigger than its n neighbours to the left and to the right.

    Hence in the sequence

    >>> x = [3, 0, 0, 4, 0, 0, 13]

    4 is a peak of support 1 and 2 because in the subsequences

    >>> [0, 4, 0]
    >>> [0, 0, 4, 0, 0]

    4 is still the highest value. Here, 4 is not a peak of support 3 because 13 is the 3th neighbour to the right of 4
    and its bigger than 4.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param n: the support of the peak
    :type n: int
    :return: the value of this feature
    :return type: float
    """
    x_reduced = x[:, n:-n, :]

    res = None
    for i in range(1, n + 1):
        result_first = x_reduced > np.roll(x, i, axis=1)[:, n:-n, :]

        if res is None:
            res = result_first
        else:
            res &= result_first

        res &= x_reduced > np.roll(x, -i, axis=1)[:, n:-n, :]
    return np.sum(res, axis=1)


def index_mass_quantile(x: np.ndarray, q: float) -> np.ndarray:
    """
    Calculates the relative index i of time series x where q% of the mass of x lies left of i.
    For example for q = 50% this feature calculator will return the mass center of the time series.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"q": x} with x float
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    abs_x = np.abs(x)
    s = np.repeat(np.sum(abs_x, axis=1)[:, np.newaxis, :], x.shape[1], axis=1)

    mass_centralized = np.cumsum(abs_x, axis=1) / s

    return (np.argmax(mass_centralized >= q, axis=1) + 1) / x.shape[1]


def _unique(x: np.ndarray) -> np.ndarray:
    return np.diff(np.sort(x, axis=1), axis=1) > 0


def _count_values(x: np.ndarray) -> np.ndarray:
    unique = _unique(x)
    return unique.sum(axis=1) + 1
