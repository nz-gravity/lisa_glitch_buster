import os

import bilby.core.result
import corner
import matplotlib.pyplot as plt
import numpy as np
from bilby import run_sampler
from eryn.ensemble import EnsembleSampler, State
from eryn.prior import ProbDistContainer, uniform_dist

from .constants import (
    NDIM,
    PARAM_LATEX,
    SCALE_RANGE,
    START_RANGE,
    TAU_RANGE,
    TIMES,
    TOBS,
    XI_RANGE,
)
from .data_generator import Data
from .logger import logger
from .postproc.image_utils import concat_images
from .postproc.plot_corner import plot_corner
from .postproc.pulse_plotter import plot_pulse

# from .backend.fisher import FisherMatrixPosteriorEstimator


def get_prior(start_time):
    t0 = max(0, start_time - 1000)
    t1 = min(TOBS, start_time + 1000)
    return ProbDistContainer(
        {
            0: uniform_dist(t0, t1),
            1: uniform_dist(*SCALE_RANGE),
            2: uniform_dist(*TAU_RANGE),
            3: uniform_dist(*XI_RANGE),
        }
    )


class GlitchFitter:
    def __init__(
        self,
        data: Data = None,
        start_time: float = None,
        label: str = None,
        seed: int = None,
        outdir: str = "outdir",
    ):
        if seed:
            np.random.seed(seed)
        if data is None:
            data = Data(seed=seed)
        self.data = data
        self.start_time = (
            data.injection_params["start"]
            if start_time is None
            else start_time
        )
        self.prior = get_prior(self.start_time)
        self.label = data.label if label is None else label
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.data.plot(outdir)

    def plot_trace(self, chain, nwalkers, fname=None):
        fig, ax = plt.subplots(NDIM, 1)
        fig.set_size_inches(10, 8)
        for i in range(NDIM):
            for walk in range(nwalkers):
                ax[i].plot(chain[:, 0, walk, :, i])
                ax[i].set_ylabel(PARAM_LATEX[i])
                # draw horizontal line at true value
                ax[i].axhline(self.data.trues[i], color="k", linestyle="--")

        fig.suptitle(f"SNR{self.data.snr:.2f}")
        fname = f"{self.outdir}/chains.png" if fname is None else fname
        plt.savefig(fname)

    def plot_corner(self, chain, fname=None):
        samples = chain.reshape(-1, NDIM)
        corner.corner(samples, truths=self.data.trues, labels=PARAM_LATEX)
        fname = f"{self.outdir}/corner.png" if fname is None else fname
        plt.savefig(fname)

    def plot_posterior(self, chain, fname=None):
        samples = chain.reshape(-1, NDIM)
        # get 100 random samples from the chain
        samples = samples[
            np.random.choice(samples.shape[0], 100, replace=False)
        ]
        signals = np.array(
            [self.data.analysis.signal_gen(*s, TIMES)[0] for s in samples]
        )
        qtls = np.percentile(signals, [0.05, 0.5, 0.95], axis=0)
        fig, ax = plt.subplots(1, 1)
        ax.plot(TIMES, qtls[1], color="C0", label="Median")
        # ax.fill_between(
        #     TIMES,
        #     qtls[0],
        #     qtls[2],
        #     color="C0",
        #     alpha=0.5,
        #     label="90% CI",
        # )
        for s in signals:
            ax.plot(TIMES, s, color="C0", alpha=0.1)
        ax.plot(
            TIMES, self.data.injection[0], color="k", label="True", ls="--"
        )
        ax.legend()
        fname = f"{self.outdir}/ppc.png" if fname is None else fname
        plt.savefig(fname)

    def run_mcmc(self, nwalkers=10, nsteps=1500, burn=1500):
        sampler = EnsembleSampler(
            nwalkers,
            NDIM,
            self.data.analysis.eryn_likelihood_function,
            priors=self.prior,
            args=(TIMES,),
        )
        start_state = State(self.prior.rvs(size=(1, nwalkers, 1)))
        sampler.run_mcmc(start_state, nsteps, progress=True, burn=burn)
        chain = sampler.get_chain()["model_0"]

        self.plot_trace(chain, nwalkers, f"{self.outdir}/tmp1.png")
        self.plot_corner(chain, f"{self.outdir}/tmp2.png")
        self.plot_posterior(chain, f"{self.outdir}/tmp3.png")
        concat_images(
            [
                f"{self.outdir}/tmp1.png",
                f"{self.outdir}/tmp2.png",
                f"{self.outdir}/tmp3.png",
            ],
            f"{self.outdir}/result.png",
        )

        # save
        np.save(f"{self.outdir}/chain.npy", chain)
        self.data.save_injection(self.outdir)

    # def get_fisher_posterior(self, n_sample=1000):
    #     fpe = FisherMatrixPosteriorEstimator(
    #         likelihood=self.likelihood,
    #         priors=self.priors,
    #         n_prior_samples=1000,
    #     )
    #     s0 = fpe.get_maximum_likelihood_sample()
    #     return fpe.sample_dataframe(s0, n_sample)

    # def compute_posterior_predictive(self, posterior, max_n_samp=1000):
    #     max_n_samp = min(max_n_samp, len(posterior))
    #     samples = posterior.sample(max_n_samp).to_dict("records")
    #     posterior_predictive = np.array(
    #         [self.model(self.times, **sample) for sample in samples]
    #     )
    #
    #     return posterior_predictive

    # def plot_corner(self, save_fn=None):
    #     if self.result is None:
    #         raise ValueError("No result found. Run the sampler first.")
    #
    #     p = self.result.posterior
    #     fisher_posterior = self.get_fisher_posterior(n_sample=len(p))
    #     params = fisher_posterior.columns
    #     p = p[params].values
    #     fisher_posterior = fisher_posterior[params].values
    #
    #     fig = plot_corner(
    #         chains=[p, fisher_posterior],
    #         chainLabels=["Sampling", "Fisher"],
    #         paramNames=params,
    #         truths=[self.injection_parameters[p] for p in params],
    #     )
    #     fig.savefig(save_fn)

    # def plot(self, save_fn=None, plot_fisher=False):
    #     pulse, posterior_predictive = None, None
    #     if self.injection_parameters is not None:
    #         pulse = self.model(self.times, **self.injection_parameters)
    #
    #     if self.result is not None:
    #         posterior_predictive = self.compute_posterior_predictive(
    #             self.result.posterior
    #         )
    #
    #     else:
    #         posterior_predictive = None
    #     ax = plot_pulse(
    #         self.amplitudes,
    #         self.times,
    #         pulse=pulse,
    #         posterior_predictive=posterior_predictive,
    #         color="C0",
    #         label="Posterior Samples",
    #     )
    #
    #     if plot_fisher:
    #         fisher_post = self.get_fisher_posterior(n_sample=1000)
    #         posterior_predictive = self.compute_posterior_predictive(fisher_post)
    #         ax = plot_pulse(
    #             self.amplitudes,
    #             self.times,
    #             pulse=pulse,
    #             posterior_predictive=posterior_predictive,
    #             ax=ax,
    #             color="C1",
    #             label="Fisher Samples",
    #         )
    #
    #     # # annnotate top right with SNR
    #     # if self.injection_snr:
    #     #     ax.annotate(
    #     #         f"SNR: {self.injection_snr:.2f}",
    #     #         xy=(0.95, 0.05),
    #     #         xycoords="axes fraction",
    #     #         horizontalalignment="right",
    #     #     )
    #     if save_fn:
    #         ax.get_figure().savefig(save_fn)

    # return ax
