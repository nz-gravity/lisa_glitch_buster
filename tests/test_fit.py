import numpy as np
import pytest

from lisa_glitch_buster.backend.model import FRED_pulse
from lisa_glitch_buster.backend.priors import get_priors
from lisa_glitch_buster.glitch_fitter import GlitchFitter


def test_basic_fit(tmpdir):

    np.random.seed(0)

    NOISE_SIG = 0.5
    N = 200
    T_START, T_END = 0, 10
    times = np.linspace(T_START, T_END, N)
    # white noise
    noise = np.random.normal(0, NOISE_SIG, N)
    # inject a FRED pulse
    true_params = get_priors(trigger_time=T_START + 2).sample(1)

    true_params = {
        "start": 2,
        "scale": 1,
        "tau": 1,
        "xi": 1,
        "sigma": NOISE_SIG,
    }
    pulse = FRED_pulse(times, **true_params)
    data = pulse + noise

    fitter = GlitchFitter(
        data=data, times=times, trigger_time=T_START + 2, model="FRED_pulse"
    )
    res = fitter.run_sampler(
        injection_parameters=true_params,
        plot=True,
        outdir=f"{tmpdir}/outdir_glitch_pe",
        # clean=True,
        sampler="dynesty",
    )
    ax = fitter.plot()
    ax.get_figure().savefig(f"{tmpdir}/glitch_fit.png")
