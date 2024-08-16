from .fred_pulse import FRED_pulse
from .sine_gaussian import sine_gaussian

MODELS = dict(FRED_pulse=FRED_pulse, sine_gaussian=sine_gaussian)
