import numpy as np


def get_snr(data, model):
    """
    Calculate the signal-to-noise ratio of a glitch.

    Parameters
    ----------
    data: array_like
        The data containing the glitch.

    model: array_like
        The model of the glitch.

    Returns
    -------
    snr: float
        The signal-to-noise ratio of the glitch.
    """
    #  time series SNR
    snr = np.inner(data, model) / np.sqrt(np.inner(model, model))
    return snr
