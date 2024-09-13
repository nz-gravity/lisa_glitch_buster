import os
import sys

from tqdm.auto import trange

from lisa_glitch_buster.glitch_fitter_eryn import GlitchFitter

if __name__ == "__main__":
    idx = int(sys.argv[1])
    for i in trange(idx * 100, (idx + 1) * 100):
        out = f"outdir/seed{i}"
        os.makedirs(out, exist_ok=True)
        g = GlitchFitter(seed=i, outdir=out)
        g.run_mcmc(nwalkers=10, nsteps=1500, burn=1500)
