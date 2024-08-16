import matplotlib.pyplot as plt
import numpy as np

from lisa_glitch_buster.backend.model import FRED_pulse, sine_gaussian


def test_FRED_pulse(tmpdir, make_plots):
    times = np.linspace(0, 10, 1000)
    start = 2
    scale = 1
    tau = 1
    xi = 1

    rate = FRED_pulse(times, start, scale, tau, xi)

    assert len(rate) == len(times)
    assert np.all(rate >= 0)
    assert np.max(rate) <= scale

    if make_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(times, rate)
        plt.title("FRED Pulse")
        plt.xlabel("Time")
        plt.ylabel("Rate")
        plt.savefig(f"{tmpdir}/FRED_pulse_plot.png")
        plt.close()


def test_sine_gaussian(tmpdir, make_plots):
    times = np.linspace(0, 10, 1000)
    res_begin = 2
    sg_A = 1
    sg_lambda = 1
    sg_omega = 2 * np.pi
    sg_phi = 0

    rate = sine_gaussian(times, res_begin, sg_A, sg_lambda, sg_omega, sg_phi)

    assert len(rate) == len(times)
    assert np.min(rate) >= -sg_A
    assert np.max(rate) <= sg_A

    if make_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(times, rate)
        plt.title("Sine-Gaussian")
        plt.xlabel("Time")
        plt.ylabel("Rate")
        plt.savefig(f"{tmpdir}/sine_gaussian_plot.png")
        plt.close()
