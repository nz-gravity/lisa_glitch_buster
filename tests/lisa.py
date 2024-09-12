import numpy as np
import matplotlib.pyplot as plt
from lisatools.utils.constants import YRSID_SI
from copy import deepcopy  # can be useful

from lisatools.sensitivity import SensitivityMatrix, LISASens, A1TDISens
from lisatools.datacontainer import DataResidualArray
from lisatools.analysiscontainer import AnalysisContainer


from lisa_glitch_buster.backend.model.fred_pulse import FRED_pulse, fred_end_time

# imports
from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.prior import uniform_dist, ProbDistContainer, log_uniform






dt = 10.0
Tobs = 50 * 60 * 60 # 50 hrs

Nobs = int(Tobs / dt)

t = np.linspace(0, Tobs, Nobs)
# just positive frequencies
f = np.fft.rfftfreq(Nobs, dt)

sens_kwargs = dict(
    stochastic_params=(1.0 * YRSID_SI,)
)

sens_mat = SensitivityMatrix(f, [A1TDISens, A1TDISens], **sens_kwargs)


INJECTION_PARAMS = dict(
    start=Tobs*0.25,
    scale=1e-20,
    tau=100,
    xi=1
)

def fred_waveform(start:float, scale:float, tau:float, xi:float, t:np.ndarray) -> np.ndarray:
    p = FRED_pulse(t, start=start, scale=scale, tau=tau, xi=xi)
    return [p,p]

injection = fred_waveform(t=t, **INJECTION_PARAMS)

plt.plot(t, injection[0], label='hplus')
plt.plot(t, injection[0], label='hcross')
# zoom in from T
plt.xlim(INJECTION_PARAMS['start'] - 10, fred_end_time(**INJECTION_PARAMS) + 10)
plt.legend()
plt.show()

simulated_data = DataResidualArray(injection, dt=dt)
fig, ax = simulated_data.loglog()
plt.show()


analysis = AnalysisContainer(
    simulated_data,
    sens_mat,
    signal_gen=fred_waveform,
)

fig, ax = analysis.loglog()
plt.show()

def compare_with_template(*waveform_args):
    opt_snr, det_snr = analysis.calculate_signal_snr(*waveform_args, source_only=False)
    lnl = analysis.calculate_signal_likelihood(*waveform_args, source_only=False)
    print(f"Optimal SNR: {opt_snr:.2f}")
    print(f"Det SNR: {det_snr:.2f}")
    print(f"Log-likelihood: {lnl:.2f}")


## TEST LNL IS LESS WHEN AWAY FROM INJECTION
compare_with_template(*INJECTION_PARAMS.values(), t)
p = INJECTION_PARAMS.copy()
p['start'] = p['start'] + -30
compare_with_template(*p.values(), t)

## TEST THAT LNL IS CENTERED AROUND INJECTION
x_vals = np.linspace(-10.0 + INJECTION_PARAMS['start'], 10.0+INJECTION_PARAMS['start'], 1000)
lnl_vals = []
for x in x_vals:
    p = INJECTION_PARAMS.copy()
    p['start'] = x
    lnl = analysis.calculate_signal_likelihood(*p.values(), t, source_only=False)
    lnl_vals.append(lnl)
plt.plot(x_vals, lnl_vals)
plt.axvline(INJECTION_PARAMS['start'], color='r')
plt.show()





def check_prior_validity(prior, n=1000):
    fig = plt.figure()
    plt.plot(t, injection[0], label='injection', color='blue', zorder=10)
    samples = prior.rvs(n)
    snrs = np.zeros(n)
    for i,s in enumerate(samples):
        signal = analysis.signal_gen(*s, t)
        plt.plot(t, signal[0],alpha=0.1,color='k')
        tend = fred_end_time(*s)
        lnl = analysis.calculate_signal_likelihood(*s, t, source_only=False)
        if tend >= Tobs:
            print(f"End time {tend} is greater than Tobs {s}")
            plt.show()
            return
        if not np.isfinite(lnl):
            print(f"Invalid sample: {s}")
            return
        _, snr = analysis.calculate_signal_snr(*s, t, source_only=True)

        # if snr<0:
        #     print(f"Invalid SNR: {snr}")
        #     plt.plot(t, signal[0],alpha=1,color='r')
        #     plt.show()
        #     return

        snrs[i] = np.abs(snr)
    plt.show()
    print("All samples valid")

    min_bin_edge = 1e-5
    collection_bin = snrs[snrs < min_bin_edge]
    adjusted_snrs = snrs[snrs >= min_bin_edge]

    # Add the collection bin count to the adjusted snrs
    adjusted_snrs = np.append(adjusted_snrs, [min_bin_edge] * len(collection_bin))

    # Define the bins
    log_snr_bins = np.geomspace(min_bin_edge, 1e4, 50)

    # Plot the histogram with the adjusted snrs
    plt.hist(adjusted_snrs, bins=log_snr_bins)

    # Mark the collection bin
    plt.axvline(min_bin_edge, color='r', linestyle='dashed', linewidth=1)
    # plt.text(min_bin_edge, plt.ylim()[1] * 0.9, 'Collection Bin', rotation=90, verticalalignment='center', color='r')
    plt.xscale('log')
    plt.show()

prior = ProbDistContainer({
    0:uniform_dist(0, Tobs/2), # start
    1:log_uniform(INJECTION_PARAMS['scale']*1e-2, INJECTION_PARAMS['scale']*1e+2),# scale
    2:uniform_dist(INJECTION_PARAMS['tau']*0.1, INJECTION_PARAMS['tau']*20), # tau
    3:log_uniform(1e-3, 10) # xi
})

print(prior)
check_prior_validity(prior, n=1000)


nwalkers = 10
sampler = EnsembleSampler(
    nwalkers,
    4,
    analysis.eryn_likelihood_function,
    prior,
    args=(t,),

)
ndim = len(INJECTION_PARAMS)
# best_guess = np.array([*INJECTION_PARAMS.values()]).reshape(-1,1)
# best_guess = np.repeat(best_guess[np.newaxis, :], nwalkers, axis=0)
# best_guess = best_guess.reshape((1, nwalkers, 1, ndim))
# start_state = State(best_guess)
start_state = State(prior.rvs(size=(1, nwalkers, 1)))
sampler.run_mcmc(start_state, 2000, progress=True)
import corner


#### Chains
fig, ax = plt.subplots(ndim, 1)
fig.set_size_inches(10, 8)
for i in range(ndim):
    for walk in range(nwalkers):
        ax[i].plot(sampler.get_chain()['model_0'][:, 0, walk, :, i])



samples = sampler.get_chain()['model_0'].reshape(-1, 4)
corner.corner(samples, truths=np.array([*INJECTION_PARAMS.values()]))
plt.show()