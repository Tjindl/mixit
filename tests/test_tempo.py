"""Choosing the mix tempo, and time-stretching without changing pitch."""

import numpy as np
import pytest

from audio_io import SAMPLE_RATE as SR
from tempo import stretch_beat_times, stretch_rate, target_bpm, time_stretch


def dominant_frequency(mono):
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    return np.argmax(spectrum) * SR / len(mono)


def test_stretch_rate_is_target_over_song_bpm():
    assert stretch_rate(120, 126) == pytest.approx(1.05)  # 5% faster
    assert stretch_rate(128, 124) == pytest.approx(0.96875)  # about 3% slower


def test_mix_tempo_is_the_median():
    assert target_bpm([120.0, 126.0, 124.0]) == 124.0
    assert target_bpm([120.0, None, 126.0]) == 123.0  # songs with no tempo are ignored
    assert target_bpm([None]) is None


@pytest.mark.parametrize("rate", [0.92, 1.08])
def test_stretching_changes_the_length_but_not_the_pitch(sine, rate):
    audio = sine(440, 2.0)
    stretched = time_stretch(audio, rate)
    assert stretched.dtype == np.float32 and stretched.shape[1] == 2
    assert len(stretched) == pytest.approx(len(audio) / rate, rel=0.002)
    assert dominant_frequency(stretched[:, 0]) == pytest.approx(440, abs=2)  # still an A


def test_peaks_above_1_survive_the_stretch(sine):
    # Loudness normalization can push peaks past 1.0. Rubber Band's 16-bit temporary
    # files would clip them flat unless time_stretch scales the audio down first.
    # (Rubber Band's output peaks a few % above its input, so check "clearly above
    # 1.0" rather than an exact value; clipped audio would top out near 1.0.)
    loud = sine(440, 2.0, amplitude=1.5)
    assert np.abs(time_stretch(loud, 1.05)).max() > 1.4


def test_stretching_keeps_left_and_right_separate(sine):
    stereo = np.column_stack([sine(440, 2.0, stereo=False), sine(660, 2.0, stereo=False)])
    stretched = time_stretch(stereo, 1.05)
    assert dominant_frequency(stretched[:, 0]) == pytest.approx(440, abs=2)
    assert dominant_frequency(stretched[:, 1]) == pytest.approx(660, abs=2)


def test_stretching_keeps_left_and_right_in_sync(click_track):
    audio, _ = click_track(120, seconds=10)
    stereo = np.column_stack([audio[:, 0], 0.5 * audio[:, 0]])  # right = left, half as loud
    left, right = time_stretch(stereo, 1.05).T
    assert np.corrcoef(left, right)[0, 1] > 0.999  # same waveform, same timing
    assert np.sqrt(np.mean(right**2) / np.mean(left**2)) == pytest.approx(0.5, rel=0.02)


def test_beats_really_move_to_t_over_rate(click_track, hit_centroids):
    """Rescaling beat times by 1/rate (instead of re-detecting) matches the stretched audio."""
    audio, true_beats = click_track(120, seconds=20)
    rate = 1.05
    before = hit_centroids(audio, true_beats * SR)
    after = hit_centroids(time_stretch(audio, rate), stretch_beat_times(true_beats, rate) * SR)
    slope, intercept = np.polyfit(before, after, 1)  # where hits were -> where they are now
    assert slope == pytest.approx(1 / rate, rel=1e-3)
    assert abs(intercept) / SR < 0.005  # no overall shift (under 5 ms)
