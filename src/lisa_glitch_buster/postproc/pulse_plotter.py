import matplotlib.pyplot as plt
import numpy as np
from ..backend.snr import get_snr

def plot_pulse(
    data,
    time,
    pulse: np.ndarray = None,
    posterior_predictive=None,
    ax=None,
    color="C0",
    label="Posterior",
):
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(time, data, label="Data", color="lightgray", zorder=-10)
    if pulse is not None:
        snr = get_snr(data, pulse)
        ax.plot(time, pulse, label=f"True (SNR={snr:.2f})", color="black", zorder=10)
    if posterior_predictive is not None:
        quntiles = np.percentile(
            posterior_predictive, [0.05, 0.5, 0.95], axis=0
        )
        ax.fill_between(time, quntiles[0], quntiles[2], color=color, alpha=0.3)
        ax.plot(time, quntiles[1], color=color, label=label)

    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Data")

    return ax
