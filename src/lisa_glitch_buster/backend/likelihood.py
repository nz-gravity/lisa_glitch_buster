from bilby.core.likelihood import GaussianLikelihood, Likelihood
from bilby.core.prior import Uniform


def get_likelihood(data, times, model, priors, noise_sigma=None):

    if noise_sigma is None:
        # then we need to estimate the noise sigma (unknown)
        priors["sigma"] = Uniform(name="sigma", minimum=1e-2, maximum=10)

    return GaussianLikelihood(
        x=times,
        y=data,
        func=model,
        priors=priors,
        sigma=noise_sigma,
    )



def ErynLikelihood(Likelihood):

    def __init__(self, lnl_fun, time, parameters={}):
        super(self).__init__()
        self.lnl_fun = lnl_fun
        self.time = time
        self.parameters = parameters

    def log_likelihood(self):
        return self.lnl_fun(*self.parameters.values(), self.time)

