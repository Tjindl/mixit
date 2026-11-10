import numpy as np
import pytest

from processing import rms_db
from transitions import equal_power_crossfade, equal_power_gains


@pytest.mark.parametrize("n", [2, 3, 101, 44_100])
def test_gains_satisfy_sin2_plus_cos2_equals_1(n):
    fade_out, fade_in = equal_power_gains(n)
    np.testing.assert_allclose(fade_out**2 + fade_in**2, 1.0, atol=1e-12)


def test_gains_go_from_full_to_silent():
    fade_out, fade_in = equal_power_gains(1000)
    assert fade_out[0] == 1.0 and fade_in[0] == 0.0
    assert fade_out[-1] == pytest.approx(0.0, abs=1e-12) and fade_in[-1] == pytest.approx(1.0)
    assert np.all(np.diff(fade_out) < 0) and np.all(np.diff(fade_in) > 0)


def test_both_gains_are_minus_3_db_halfway():
    fade_out, fade_in = equal_power_gains(101)  # odd length, so index 50 is exactly t = 0.5
    assert fade_out[50] == pytest.approx(1 / np.sqrt(2))
    assert fade_in[50] == pytest.approx(1 / np.sqrt(2))
    assert 20 * np.log10(fade_in[50]) == pytest.approx(-3.01, abs=0.01)


def test_crossfade_multiplies_each_chunk_by_its_gain():
    a = np.full((1000, 2), 0.5, dtype=np.float32)
    b = np.full((1000, 2), 0.25, dtype=np.float32)
    fade_out, fade_in = equal_power_gains(1000)
    blended = equal_power_crossfade(a, b)
    assert blended.shape == (1000, 2) and blended.dtype == np.float32
    for channel in (0, 1):
        np.testing.assert_allclose(blended[:, channel], 0.5 * fade_out + 0.25 * fade_in, rtol=1e-6)


def test_loudness_stays_steady_through_the_crossfade():
    """The reason for sin/cos. Two unrelated signals at the same loudness (random
    noise stands in for two songs) crossfade without a dip in the middle."""
    rng = np.random.default_rng(seed=0)
    n = 110_000
    a = (0.1 * rng.standard_normal((n, 2))).astype(np.float32)
    b = (0.1 * rng.standard_normal((n, 2))).astype(np.float32)

    blended = equal_power_crossfade(a, b)
    # Loudness of 11 consecutive slices of the crossfade, relative to the input level:
    levels = [rms_db(chunk) - rms_db(a) for chunk in np.array_split(blended, 11)]
    np.testing.assert_allclose(levels, 0.0, atol=0.3)  # within 0.3 dB the whole way

    # For contrast: a linear crossfade (gains 1 - t and t) sags by 3 dB in the middle.
    t = np.linspace(0.0, 1.0, n)[:, np.newaxis]
    linear = a * (1 - t) + b * t
    middle = np.array_split(linear, 11)[5]
    assert rms_db(middle) - rms_db(a) == pytest.approx(-3.0, abs=0.3)


def test_chunks_must_be_the_same_length():
    with pytest.raises(ValueError):
        equal_power_crossfade(np.zeros((10, 2)), np.zeros((9, 2)))
