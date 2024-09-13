from lisa_glitch_buster.data_generator import Data
from lisa_glitch_buster.glitch_fitter_eryn import GlitchFitter
from lisa_glitch_buster.injection_generator import InjectionGenerator


def test_draw_injection(tmpdir):
    fig, ax = InjectionGenerator.plot_prior_samples(n=100)
    fig.savefig(f"{tmpdir}/prior_samples.png")


def test_datagen(tmpdir):
    d = Data(seed=42)
    d.plot(tmpdir)
    assert d.snr > 8


def test_pe(tmpdir):
    g = GlitchFitter(
        seed=42,
    )
    g.run_mcmc(nwalkers=10, nsteps=1500, burn=1500)

    #
    #
    # def plot_lnl_vs_start(self, save_path=None):
    #     x_vals = np.linspace(-10.0 + self.injection_params['start'], 10.0 + self.injection_params['start'], 1000)
    #     lnl_vals = []
    #     for x in x_vals:
    #         p = self.injection_params.copy()
    #         p['start'] = x
    #         lnl = self.analysis.calculate_signal_likelihood(*p.values(), self.t, source_only=False)
    #         lnl_vals.append(lnl)
    #     plt.plot(x_vals, lnl_vals)
    #     plt.axvline(self.injection_params['start'], color='r')
    #     if save_path:
    #         plt.savefig(save_path)
    #     plt.show()
