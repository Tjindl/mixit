"""Stage 4 transitions: the 3-band EQ, the bass swap, and the echo out."""

import numpy as np
import pytest

from audio_io import SAMPLE_RATE as SR
from processing import rms
from transitions import (
    bass_swap_blend,
    bass_swap_gains,
    echo_join,
    echo_tail,
    lowpass_sweep,
    three_band,
)


def tone(freq, seconds, amplitude=0.5):
    t = np.arange(round(seconds * SR)) / SR
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    return np.column_stack([wave, wave]).astype(np.float32)


def envelope(audio, window=441):
    """Loudness over time: RMS of each 10 ms window (left channel)."""
    x = audio[: len(audio) // window * window, 0].astype(np.float64)
    return np.sqrt(np.mean(x.reshape(-1, window) ** 2, axis=1))


def test_the_three_bands_split_by_frequency_and_add_back_up_exactly():
    mixture = tone(60, 2.0) + tone(1000, 2.0) + tone(8000, 2.0)
    low, mid, high = three_band(mixture, SR)
    np.testing.assert_allclose(low + mid + high, mixture, atol=1e-6)
    middle = slice(SR // 2, 3 * SR // 2)  # away from the ends
    assert rms(low[middle]) == pytest.approx(rms(tone(60, 1.0)), rel=0.05)  # the 60 Hz tone, alone
    assert rms(mid[middle]) == pytest.approx(rms(tone(1000, 1.0)), rel=0.05)
    assert rms(high[middle]) == pytest.approx(rms(tone(8000, 1.0)), rel=0.05)


def test_the_two_basslines_are_never_both_at_full_level():
    """Bass swap: A's bass plays until the downbeat halfway, B's after it, never both at full."""
    swap_at, window = 4 * SR, 882  # 20 ms windows: exactly 1 and 2 cycles of the two basslines
    bass_a, bass_b = tone(50, 8.0), tone(100, 8.0)  # two basslines, A and B
    silence = np.zeros_like(bass_a)
    # The blend is linear in each song, so blending one with silence shows that song's part alone.
    a_part = envelope(bass_swap_blend(bass_a, silence, swap_at, SR), window) / rms(bass_a)
    b_part = envelope(bass_swap_blend(silence, bass_b, swap_at, SR), window) / rms(bass_b)
    assert np.max(np.minimum(a_part, b_part)) < 0.75  # never both near full (at most -3 dB each, mid-swap)
    both_audible = np.flatnonzero((a_part > 0.1) & (b_part > 0.1))
    assert 0 < len(both_audible) <= 2  # only during the 20 ms swap
    k = swap_at // window  # the window where the swap ends
    assert np.all(a_part[: k - 2] > 0.97) and np.all(b_part[: k - 2] < 0.03)  # A's bass, B's cut
    assert np.all(a_part[k + 1 :] < 0.03) and np.all(b_part[k + 1 :] > 0.97)  # then the other way round


def test_b_s_bass_is_fully_in_on_the_downbeat_itself():
    low_out, low_in = bass_swap_gains(1000, swap_at=600, fade=50)
    assert low_in[599] == pytest.approx(1.0, abs=0.01) and low_out[599] == pytest.approx(0.0, abs=0.04)
    assert low_in[549] == 0.0 and low_out[549] == 1.0
    np.testing.assert_allclose(low_out**2 + low_in**2, 1.0)  # equal power through the swap


def test_mids_and_highs_crossfade_smoothly_across_the_whole_blend():
    a, b = tone(1000, 4.0), tone(1500, 4.0)
    blended = bass_swap_blend(a, b, 2 * SR, SR)
    levels = envelope(blended) / rms(a)
    assert np.all(np.abs(levels[20:-20] - 1.0) < 0.15)  # steady loudness (equal power), no dip or bump
    assert envelope(bass_swap_blend(a, np.zeros_like(b), 2 * SR, SR))[-5:].max() < 0.01  # A gone by the end


def test_the_echo_decays_geometrically():
    send = tone(440, 0.5, amplitude=0.8)  # "A's last beat": half a second at 120 BPM
    delay, feedback = len(send), 0.55
    tail = echo_tail(send, 8 * delay, SR, feedback=feedback, sweep=None)
    peaks = [np.abs(tail[k * delay + 1000 : (k + 1) * delay - 1000]).max() for k in range(6)]
    for k, level in enumerate(peaks):
        assert level == pytest.approx(0.8 * feedback ** (k + 1), rel=0.02)  # each repeat = previous x feedback


def test_the_echo_does_not_clip_and_fades_to_silence():
    send = tone(440, 0.5, amplitude=0.95)
    tail = echo_tail(send, 8 * len(send), SR)  # with the closing low-pass sweep
    assert np.abs(tail).max() < 0.95  # quieter than what it echoes, so it can't clip
    repeats = [rms(tail[k * len(send) : (k + 1) * len(send)]) for k in range(8)]
    assert all(later < earlier for earlier, later in zip(repeats, repeats[1:]))  # always fading
    assert np.abs(tail[-10:]).max() < 1e-3  # and silent at the very end: no click when it stops


def test_the_lowpass_sweep_closes_over_time():
    noise = (0.3 * np.random.default_rng(0).standard_normal((2 * SR, 2))).astype(np.float32)
    swept = lowpass_sweep(noise, SR, 8000.0, 300.0)
    first, last = swept[: SR // 4], swept[-SR // 4 :]

    def treble(x):  # energy above 2 kHz
        spectrum = np.abs(np.fft.rfft(x[:, 0])) ** 2
        return spectrum[np.fft.rfftfreq(len(x), 1 / SR) > 2000].sum()

    assert treble(last) < 0.01 * treble(first)


def test_echo_join_brings_b_in_without_a_click():
    tail, b = np.zeros((2000, 2), np.float32), np.ones((2000, 2), np.float32)
    joined = echo_join(tail, b, SR)
    assert joined[0, 0] == 0.0 and np.all(np.diff(joined[:300, 0]) >= 0) and joined[-1, 0] == 1.0
