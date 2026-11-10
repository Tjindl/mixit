import numpy as np
import pytest

from audio_io import SAMPLE_RATE as SR
from processing import (
    FRAME_LENGTH,
    clipping_gain,
    db_to_gain,
    normalize_loudness,
    prevent_clipping,
    rms,
    rms_db,
    trim_silence,
)


def silence(seconds):
    return np.zeros((round(seconds * SR), 2), dtype=np.float32)


def test_rms_of_a_sine_is_its_amplitude_over_root_2(sine):
    assert rms(sine(440, 1.0, amplitude=0.8)) == pytest.approx(0.8 / np.sqrt(2), rel=1e-4)


def test_decibel_conversions():
    assert db_to_gain(0) == 1.0
    assert db_to_gain(-20) == pytest.approx(0.1)
    assert db_to_gain(-6.02) == pytest.approx(0.5, rel=1e-3)


def test_normalize_brings_quiet_and_loud_tracks_to_the_same_level(sine):
    quiet, loud = sine(440, 2.0, amplitude=0.02), sine(220, 2.0, amplitude=0.9)
    for track in (quiet, loud):
        assert rms_db(normalize_loudness(track, target_db=-16.0)) == pytest.approx(-16.0, abs=1e-3)


def test_normalizing_silence_leaves_it_alone():
    np.testing.assert_array_equal(normalize_loudness(silence(0.1)), silence(0.1))


def test_trim_removes_silence_at_both_ends_but_keeps_all_the_music(sine):
    music = sine(440, 2.0)
    trimmed = trim_silence(np.concatenate([silence(1.0), music, silence(1.5)]))
    # Trimming works in whole frames, so up to one frame of silence can remain at each end...
    assert len(music) <= len(trimmed) <= len(music) + 2 * FRAME_LENGTH
    # ...but no music is lost: all of its energy is still there.
    assert np.sum(np.square(trimmed, dtype=np.float64)) == pytest.approx(
        np.sum(np.square(music, dtype=np.float64))
    )


def test_trim_treats_a_faint_noise_floor_as_silence(sine):
    rng = np.random.default_rng(seed=1)
    hiss = (0.0005 * rng.standard_normal((SR, 2))).astype(np.float32)  # about 57 dB below the music
    music = sine(440, 2.0, amplitude=0.5)
    trimmed = trim_silence(np.concatenate([hiss, music, hiss]))
    assert len(music) <= len(trimmed) <= len(music) + 2 * FRAME_LENGTH


def test_trim_keeps_quiet_passages_in_the_middle(sine):
    audio = np.concatenate([sine(440, 1.0), silence(1.0), sine(440, 1.0)])
    assert len(trim_silence(audio)) == len(audio)


def test_trimming_pure_silence_leaves_nothing():
    assert len(trim_silence(silence(1.0))) == 0


def test_prevent_clipping_scales_the_peak_down_to_the_ceiling():
    loud = np.array([[0.5, -1.5], [1.2, 0.3]], dtype=np.float32)
    safe = prevent_clipping(loud, ceiling_db=-1.0)
    # Every sample is scaled by the same amount, so the waveform's shape is unchanged.
    np.testing.assert_allclose(safe, loud * (db_to_gain(-1.0) / 1.5), rtol=1e-6)
    assert np.abs(safe).max() <= 1.0


def test_prevent_clipping_leaves_quiet_audio_alone(sine):
    quiet = sine(440, 0.1, amplitude=0.5)
    assert clipping_gain(quiet) == 1.0
    np.testing.assert_array_equal(prevent_clipping(quiet), quiet)
