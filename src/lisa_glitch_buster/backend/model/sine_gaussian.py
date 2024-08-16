import math
import sys

import numpy as np
import scipy.special as special
from scipy.signal import convolve

from .common import MAX_EXP, MAX_FLOAT, MIN_FLOAT


def sine_gaussian(times, res_begin, sg_A, sg_lambda, sg_omega, sg_phi):
    r"""
    The sine-gaussian residual function. This pulse is not amplitude-normalised.

    .. math::

        \text{res}(t)= A_\text{res} \exp \left[
        - \left(\frac{t-\Delta_\text{res}} {\lambda_\text{res}}\right)^2
        \right] \cos\left(\omega t + \varphi \right)


    Parameters
    ----------
    times : array_like
        The input time array.
    res_begin : float
        The start time of the pulse.
    sg_A : float
        The amplitude of the pulse.
    sg_lambda : float
        The duration of the pulse.
    sg_omega : float
        The angular frequency of the cosine function.
    sg_phi: float
        The phase of the cosine function.

    Returns
    -------
    rate : ndarray
         Output array containing the residual.

    """
    s = np.exp(-np.square((times - res_begin) / sg_lambda)) * np.cos(
        sg_omega * times + sg_phi
    )
    return sg_A * s
