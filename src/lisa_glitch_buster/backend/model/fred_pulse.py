import math
import sys

import numba
import numpy as np
import scipy.special as special
from scipy.optimize import root_scalar
from scipy.signal import convolve

from .common import MAX_EXP, MAX_FLOAT, MIN_FLOAT


@numba.jit(nopython=True)
def FRED_pulse(times, start, scale, tau, xi):
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


def fred_amplitude(t, start, scale, tau, xi):
    return scale * np.exp(
        -xi * ((tau / (t - start)) + ((t - start) / tau) - 2)
    )


def fred_end_time(start, scale, tau, xi, threshold=0.01):
    a = lambda t: fred_amplitude(t, start, scale, tau, xi) - threshold * scale
    result = root_scalar(lambda t: a(t), x0=start + tau, x1=start + 10 * tau)
    return result.root


@numba.jit(nopython=True)
def waveform(
    start: float, scale: float, tau: float, xi: float, t: np.ndarray
) -> np.ndarray:
    p = FRED_pulse(t, start=start, scale=scale, tau=tau, xi=xi)
    return [p, p]
