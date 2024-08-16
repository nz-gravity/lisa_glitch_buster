from pathlib import Path

import matplotlib
from matplotlib import font_manager

HERE = Path(__file__).parent

rc_file = HERE / "matplotlibrc"
font_file = HERE / "LiberationSans.ttf"

font_file = Path(font_file)
if font_file.exists():
    font_manager.fontManager.addfont(str(font_file))
else:
    print(f"Font file not found: {font_file}")

matplotlib.rc_file(rc_file)
