import math
import sys

import numpy as np
import scipy.special as special
from scipy.signal import convolve

from .common import MAX_EXP, MAX_FLOAT, MIN_FLOAT


def FRED_pulse(times, start, scale, tau, xi, **kwargs):
    r"""
    The amplitude-normalised equation for a fast-rise expontential-decay pulse.
    The rate is calculated at each input time and returned as an array.

    .. math::

        S(t|A,\Delta,\tau,\xi) = A \exp \left[ - \xi \left(
        \frac{t - \Delta}{\tau} + \frac{\tau}{t-\Delta}  \right) -2  \right]


    Parameters
    ----------
    times : array_like
        The input time array.
    start : float
        The start time of the pulse.
    scale : float
        The amplitude of the pulse.
    tau : float
        The duration of the pulse.
    xi : float
        The asymmetry of the pulse.

    Returns
    -------
    rate : ndarray
         Output array containing the pulse.

    """

    rate = np.where(
        times - start <= 0,
        MIN_FLOAT,
        scale
        * np.exp(
            -xi
            * (
                (
                    tau
                    / np.where(
                        times - start <= 0,
                        times - start - MIN_FLOAT,
                        times - start + MIN_FLOAT,
                    )
                )
                + ((times - start) / (tau + MIN_FLOAT))
                - 2.0
            )
        ),
    )
    return rate
