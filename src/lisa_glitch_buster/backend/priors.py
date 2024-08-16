import numpy as np
from bilby.core.prior import LogUniform, PriorDict, Uniform


def get_priors(trigger_time, delta_t=1, model="FRED_pulse"):
    """
    Returns a PriorDict containing priors for the model parameters.

    Parameters
    ----------
    trigger_time: float
        The trigger time of the glitch.

    model: str
        The name of the model to use.

    Returns
    -------
    priors: PriorDict
        A PriorDict containing the priors for the model parameters.
    """

    priors = PriorDict()

    if model == "FRED_pulse":
        priors["start"] = Uniform(
            name="start",
            minimum=trigger_time - delta_t,
            maximum=trigger_time + delta_t,
        )
        priors["scale"] = LogUniform(name="scale", minimum=0.1, maximum=10)
        priors["tau"] = LogUniform(name="tau", minimum=0.1, maximum=10)
        priors["xi"] = Uniform(name="xi", minimum=1e-2, maximum=10)

    return priors
