import os
import warnings
from copy import deepcopy
from typing import Dict

import corner
import matplotlib.pyplot as plt
import numpy as np
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, log_uniform, uniform_dist
from eryn.state import State
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.utils.constants import YRSID_SI
from matplotlib.gridspec import GridSpec

from lisa_glitch_buster.backend.model.fred_pulse import fred_end_time, waveform

from .constants import DT, FREQS, SENSITIVITY_MATRIX, TIMES, TOBS, N
from .injection_generator import InjectionGenerator
from .postproc.image_utils import concat_images
from .postproc.plot_collection_hist import hist_collection


class Data:
    def __init__(self, seed=None, Tobs=TOBS, dt=DT):
        self.seed = seed
        self.ndim = len(self.injection_params)
        self.injection = waveform(**self.injection_params, t=TIMES)
        self.simulated_data = DataResidualArray(self.injection, dt=DT)
        self.analysis = AnalysisContainer(
            self.simulated_data, SENSITIVITY_MATRIX, signal_gen=waveform
        )
        self.snr = InjectionGenerator.optimal_snr(self.injection)
        print("Injected SNR: ", self.snr)

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value: int):
        self._seed = value
        np.random.seed(value)

    @property
    def injection_params(self):
        if not hasattr(self, "_injection_params"):
            if self.seed:
                self._injection_params = InjectionGenerator.draw_injection()
            else:
                self._injection_params = {
                    "start": self.Tobs * 0.25,
                    "scale": 1e-20,
                    "tau": 100,
                    "xi": 1,
                }
        return self._injection_params

    @property
    def label(self):
        return f"inj[{self.seed}]" if self.seed else "inj[default]"

    def plot_injection(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.plot(TIMES, self.injection[0], label="hplus")
        ax.plot(TIMES, self.injection[1], label="hcross")
        ax.set_xlim(
            self.injection_params["start"] - 10,
            fred_end_time(**self.injection_params) + 10,
        )
        ax.legend()
        return ax

    def plot(self, outdir):
        fig, ax = self.analysis.loglog()
        fig.suptitle(f"SNR: {self.snr:.2f}")
        ax[0].set_ylabel("Characteristic Strain")
        fig.text(0.5, 0.04, "Frequency [Hz]", ha="center")

        fig2, ax = plt.subplots(1, 1)
        ax.plot(TIMES, self.injection[0], label="Injection")
        ax.set_xlim(
            self.injection_params["start"] - 10,
            fred_end_time(**self.injection_params) + 10,
        )
        ax.legend()

        fig.savefig(f"tmp1.png")
        fig2.savefig(f"tmp2.png")
        concat_images(["tmp1.png", "tmp2.png"], f"{outdir}/data.png")

    def trues(self):
        return [*self.injection_params.values()]
