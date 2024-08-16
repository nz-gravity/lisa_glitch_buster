import os

import numpy as np
from bilby import run_sampler
from bilby.core.fisher import FisherMatrixPosteriorEstimator

from .backend.likelihood import get_likelihood
from .backend.model import MODELS
from .backend.priors import get_priors
from .postproc.pulse_plotter import plot_pulse


class GlitchFitter:
    def __init__(
        self,
        data: np.ndarray,
        times: np.ndarray,
        trigger_time: float,
        outdir: str = "outdir",
        injection_parameters: dict = None,
        model="FRED_pulse",
    ):
        self.amplitudes = data
        self.times = times
        self.trigger_time = trigger_time
        self.outdir = outdir
        self.model_name = model
        self.model = MODELS[model]
        self.injection_parameters = injection_parameters

        os.makedirs(outdir, exist_ok=True)

        self.__bayesian_setup()

    def __bayesian_setup(self):
        self.priors = get_priors(self.trigger_time, model=self.model_name)
        self.likelihood = get_likelihood(
            data=self.amplitudes,
            times=self.times,
            model=self.model,
            priors=self.priors,
        )
        self.result = None

    def run_sampler(self, **kwargs):

        kwargs["outdir"] = kwargs.get("outdir", self.outdir)
        kwargs["label"] = kwargs.get("label", "glitch_fit")
        kwargs["injection_parameters"] = kwargs.get(
            "injection_parameters", self.injection_parameters
        )
        kwargs["sampler"] = kwargs.get("sampler", "emcee")

        if kwargs["sampler"] == "dynesty":
            if "nlive" not in kwargs:
                kwargs["nlive"] = 250

        if kwargs["sampler"] == "emcee":
            if "nwalkers" not in kwargs:
                kwargs["nwalkers"] = 10
            if "nsteps" not in kwargs:
                kwargs["nsteps"] = 2000

        self.plot(save_fn=f"{self.outdir}/data.png")
        self.result = run_sampler(
            likelihood=self.likelihood, priors=self.priors, **kwargs
        )
        return self.result

    def get_fisher_posterior(self, n_sample=1000):
        fpe = FisherMatrixPosteriorEstimator(
            likelihood=self.likelihood, priors=self.priors
        )
        s0 = fpe.get_maximum_likelihood_sample()
        return fpe.sample_dataframe(s0, n_sample)

    def compute_posterior_predictive(self, posterior, max_n_samp=1000):
        max_n_samp = min(max_n_samp, len(posterior))
        samples = posterior.sample(max_n_samp).to_dict("records")
        posterior_predictive = np.array(
            [self.model(self.times, **sample) for sample in samples]
        )

        return posterior_predictive

    def plot(self, save_fn=None):
        pulse, posterior_predictive = None, None
        if self.injection_parameters is not None:
            pulse = self.model(self.times, **self.injection_parameters)

        if self.result is not None:
            posterior_predictive = self.compute_posterior_predictive(
                self.result.posterior
            )
        else:
            posterior_predictive = None
        ax = plot_pulse(
            self.amplitudes,
            self.times,
            pulse=pulse,
            posterior_predictive=posterior_predictive,
        )

        if save_fn:
            ax.get_figure().savefig(save_fn)

        return ax
