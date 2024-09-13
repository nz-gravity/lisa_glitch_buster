class GlitchAnalyser:
    def __init__(
        self, Tobs=TOBS, dt=10.0, injection_params=None, random_seed=None
    ):
        self.Tobs = Tobs
        self.dt = dt
        self.Nobs = int(Tobs / dt)
        self.t = np.linspace(0, Tobs, self.Nobs)
        self.f = np.fft.rfftfreq(self.Nobs, dt)

        self.sens_mat = SensitivityMatrix(
            self.f,
            sens_mat=[A1TDISens, A1TDISens],
            stochastic_params=(1.0 * YRSID_SI,),
        )

        self.injection_params = injection_params or {
            "start": Tobs * 0.25,
            "scale": 1e-20,
            "tau": 100,
            "xi": 1,
        }
        self.ndim = len(self.injection_params)
        if random_seed is not None:
            np.random.seed(random_seed)
            self.injection_params = self.draw_injection()
            self.label = f"inj{random_seed}"

        self.outdir = f"outdir/{self.label}"
        os.makedirs(self.outdir, exist_ok=True)

        self.injection = self.fred_waveform(t=self.t, **self.injection_params)
        self.simulated_data = DataResidualArray(self.injection, dt=self.dt)
        self.analysis = AnalysisContainer(
            self.simulated_data, self.sens_mat, signal_gen=self.fred_waveform
        )
        self.snr = self.analysis.calculate_signal_snr(
            *self.injection_params.values(), self.t, source_only=True
        )[0]
        print("Injected SNR: ", self.snr)

    def plot_analysis(self):
        fig, ax = self.analysis.loglog()
        fig.suptitle(f"SNR: {self.snr:.2f}")
        plt.savefig(f"{self.outdir}/data.png")

        fig2 = plt.figure()
        plt.plot(self.t, self.injection[0], label="hplus")
        plt.plot(self.t, self.injection[1], label="hcross")
        plt.xlim(
            self.injection_params["start"] - 10,
            fred_end_time(**self.injection_params) + 10,
        )
        plt.legend()
        plt.savefig(f"{self.outdir}/injection.png")

        # combine the two saved figures into one

    def compare_with_template(self, *waveform_args):
        opt_snr, det_snr = self.analysis.calculate_signal_snr(
            *waveform_args, source_only=False
        )
        lnl = self.analysis.calculate_signal_likelihood(
            *waveform_args, source_only=False
        )
        print(f"Optimal SNR: {opt_snr:.2f}")
        print(f"Det SNR: {det_snr:.2f}")
        print(f"Log-likelihood: {lnl:.2f}")

    def create_prior(self):
        inj_vals = [*self.injection_params.values()]

        t0 = max(0, inj_vals[0] - 1000)
        t1 = min(self.Tobs, inj_vals[0] + 1000)

        return ProbDistContainer(
            {
                0: uniform_dist(t0, t1),
                1: log_uniform(1e-20, 1e-22),
                2: log_uniform(1, 100),
                3: log_uniform(1e-3, 10),
            }
        )


def check_prior_validity(prior, t, n=1000, save_path=None):
    samples = prior.rvs(n)
    snrs = np.zeros(n)
    for i, s in enumerate(samples):
        signal = self.analysis.signal_gen(*s, t)
        plt.plot(self.t, signal[0], alpha=0.1, color="k")
        tend = fred_end_time(*s)
        lnl = self.analysis.calculate_signal_likelihood(
            *s, self.t, source_only=False
        )
        if tend >= self.Tobs:
            print(f"End time {tend} is greater than Tobs {s}")
            plt.show()
            return
        if not np.isfinite(lnl):
            print(f"Invalid sample: {s}")
            return
        _, snr = self.analysis.calculate_signal_snr(
            *s, self.t, source_only=True
        )
        snrs[i] = np.abs(snr)
    if save_path:
        plt.savefig(f"{save_path}_signals.png")
    plt.show()
    print("All samples valid")

    if save_path:
        plt.savefig(f"{save_path}_snr_hist.png")
    plt.show()


from tqdm.auto import trange

if __name__ == "__main__":
    for i in trange(100):
        GlitchAnalyser(random_seed=i).run_mcmc()
