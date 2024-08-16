import os

import numpy as np
import pytest

np.random.seed(42)


HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = f"{HERE}/out_test"


@pytest.fixture
def tmpdir() -> str:
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR, exist_ok=True)
    return TEST_DIR


@pytest.fixture
def make_plots() -> bool:
    return True
