import math
import sys

import numpy as np
import scipy.special as special
from scipy.signal import convolve

MIN_FLOAT = sys.float_info[3]
MAX_FLOAT = sys.float_info[0]
MAX_EXP = np.log(MAX_FLOAT)
