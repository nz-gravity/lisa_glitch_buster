import warnings

import numpy as np
import matplotlib.pyplot as plt
from lisatools.utils.constants import YRSID_SI
from copy import deepcopy

from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer
from lisa_glitch_buster.backend.model.fred_pulse import waveform, fred_end_time
from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.prior import uniform_dist, ProbDistContainer, log_uniform
import os
import corner
from typing import Dict
from .constants import TOBS, DT, N, TIMES, FREQS, SENSITIVITY_MATRIX, START_RANGE, SCALE_RANGE, TAU_RANGE, XI_RANGE, PARAM_NAMES
from .postproc.plot_collection_hist import hist_collection

INJ_PRIOR = ProbDistContainer({
    0: uniform_dist(*START_RANGE),
    1: uniform_dist(*SCALE_RANGE),
    2: uniform_dist(*TAU_RANGE),
    3: uniform_dist(*XI_RANGE)
})




class InjectionGenerator:
    @staticmethod
    def draw_injection(min_snr=8) -> Dict[str, float]:
        tend = TOBS
        snr = 0
        while tend >= TOBS and snr < min_snr:
            sample = dict(zip(PARAM_NAMES, *INJ_PRIOR.rvs()))
            tend = fred_end_time(**sample)
            snr = InjectionGenerator.optimal_snr(waveform(**sample, t=TIMES))
        return sample

    @staticmethod
    def optimal_snr(signal, sens_mat=SENSITIVITY_MATRIX):
        hf = DataResidualArray(signal, dt=DT)
        h_inner = 4 * hf.df * np.real(
            np.sum(hf[:, 1:].conj() * hf[:, 1:] / sens_mat[:, 1:])
        )
        return np.abs(np.sqrt(h_inner))

    @staticmethod
    def plot_prior_samples(n=1000):
        fig, axes = plt.subplots(3, 1, figsize=(3, 6))

        s = INJ_PRIOR.rvs(n)
        snrs = np.zeros(n)
        tends = np.zeros(n)
        for i, s in enumerate(s):
            signal = waveform(*s, t=TIMES)
            tends[i] = fred_end_time(*s)
            snrs[i] = InjectionGenerator.optimal_snr(signal)

            axes[0].plot(TIMES, signal[0], alpha=0.05, color='k')
            # add little notch long axis to indicate end time
            axes[0].plot([tends[i], tends[i]], [0, 0], color='r', alpha=0.05)

        hist, _ = hist_collection(snrs, bins=np.geomspace(5, 300, 50), ax=axes[1])
        if (hist[0] + hist[-1]) > 0:
            warnings.warn("SNR distribution goes beyond the desired limits...")
        axes[1].set_xscale('log')


        hist, _ = hist_collection(tends, np.linspace(0, TOBS, 50), ax=axes[2])
        if (hist[0] + hist[-1]) > 0:
            warnings.warn("End time distribution goes beyond the desired limits...")

        axes[0].set_ylabel('Strain')
        axes[0].set_xlabel('Time [s]')
        axes[1].set_ylabel('SNRs counts')
        axes[2].set_ylabel('T1 counts')

        plt.tight_layout()
        return fig, axes