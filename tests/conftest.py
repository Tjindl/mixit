"""Shared test helpers. The tests use generated sine waves, so no real songs are needed."""

import numpy as np
import pytest

from audio_io import SAMPLE_RATE


def make_sine(freq=440.0, seconds=1.0, amplitude=0.5, sr=SAMPLE_RATE, stereo=True):
    """A pure tone: float32, shape (n, 2), or (n,) if stereo=False."""
    t = np.arange(round(seconds * sr)) / sr  # the time, in seconds, of each sample
    wave = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.column_stack([wave, wave]) if stereo else wave


@pytest.fixture
def sine():
    """Gives a test the make_sine function: `sine(440, seconds=2, amplitude=0.3)`."""
    return make_sine
