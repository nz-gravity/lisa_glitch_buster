import numpy as np
from lisatools.datacontainer import DataResidualArray
from lisatools.sensitivity import A1TDISens, SensitivityMatrix
from lisatools.utils.constants import YRSID_SI

from .backend.model.fred_pulse import waveform

TOBS = YRSID_SI / 365 / 24  # 1 HR
DT = 10.0
DF = 1.0 / TOBS
N = int(TOBS / DT)
TIMES = np.linspace(0, TOBS, N)
FREQS = np.fft.rfftfreq(N, DT)

# PARAMTER RANGES
START_RANGE = (0, TOBS / 2)
SCALE_RANGE = (5e-20, 1e-22)
TAU_RANGE = (1, 100)
XI_RANGE = (1e-3, 10)
PARAM_NAMES = ["start", "scale", "tau", "xi"]
PARAM_LATEX = [r"$\Delta t$", r"$A$", r"$\tau$", r"$\xi$"]
NDIM = len(PARAM_NAMES)

SENSITIVITY_MATRIX = SensitivityMatrix(
    FREQS, sens_mat=[A1TDISens, A1TDISens], stochastic_params=(TOBS,)
)

# ENSURE CORRECT FREQUENCY ARRAY
template = DataResidualArray(waveform(1, 1, 1, 1, TIMES), dt=DT)
SENSITIVITY_MATRIX.update_frequency_arr(template.f_arr)
