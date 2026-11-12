"""Transitions: how one track hands over to the next.

A transition is any function with this shape:

    blended = transition(outgoing, incoming)

`outgoing` is the last N samples of the track that's ending, `incoming` is the
first N samples of the track that's starting (both shape (N, 2)), and `blended` is
the N samples the listener hears while both are playing.

Stage 1 had one transition, the equal-power crossfade. Stage 4 adds DJ transitions
with the same shape (see "Stage 4: DJ transitions" below): a blend with a bass
swap, a drop swap, and an echo out.
"""

from collections.abc import Callable
from functools import partial

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

Transition = Callable[[np.ndarray, np.ndarray], np.ndarray]

# Why sine and cosine? The "equal-power" idea
# -------------------------------------------
# How loud something sounds depends on its POWER, and power is proportional to
# amplitude squared. (Double the amplitude -> 4x the power -> about +6 dB.)
#
# During a crossfade we play   mix = g_out * A + g_in * B
# where A is the outgoing song, B the incoming one, and g_out, g_in their gains.
#
# Squaring that gives  g_out^2 * A^2  +  g_in^2 * B^2  +  2 * g_out * g_in * A * B.
# Two different songs are "uncorrelated": their waveforms rise and fall
# independently, so the last term, A * B, is positive as often as it's negative
# and averages out to about zero. What's left says the POWERS add:
#
#     power(mix) = g_out^2 * power(A) + g_in^2 * power(B)
#
# We loudness-normalized every song, so power(A) = power(B) = P, and
#
#     power(mix) = (g_out^2 + g_in^2) * P
#
# For the volume to stay steady through the whole crossfade we need
#
#     g_out^2 + g_in^2 = 1        at every moment.
#
# That's exactly the identity cos(x)^2 + sin(x)^2 = 1. So we sweep an angle x from
# 0 to 90 degrees (0 to pi/2 radians) and use g_out = cos(x), g_in = sin(x):
#
#      t     angle   g_out = cos   g_in = sin   g_out^2 + g_in^2
#     0.0     0 deg     1.000         0.000           1.0
#     0.5    45 deg     0.707         0.707           1.0   <- each at -3 dB
#     1.0    90 deg     0.000         1.000           1.0
#
# Picture the point (g_out, g_in) moving along a quarter of a circle of radius 1:
# its distance from the centre (the total power) never changes.
#
# Compare a LINEAR crossfade, g_out = 1 - t and g_in = t. Halfway, both are 0.5, so
# the power is 0.25 + 0.25 = 0.5: a 3 dB dip, and you hear the music sag in the
# middle of every transition. tests/test_transitions.py measures both.
#
# A caveat for later stages: if the two signals are CORRELATED (say, beat-matched
# kick drums hitting at exactly the same moment), their amplitudes add instead of
# their powers, and an equal-power fade bulges by +3 dB in the middle. That's one
# reason DJ mixers offer different crossfader curves. Worth revisiting in Stage 2.


def equal_power_gains(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the (fade_out, fade_in) gain curves for an n-sample equal-power crossfade.

    fade_out falls 1 -> 0 as cos(pi*t/2) while fade_in rises 0 -> 1 as sin(pi*t/2),
    with t running from 0 at the first sample to 1 at the last. At every sample,
    fade_out**2 + fade_in**2 == 1.
    """
    t = np.linspace(0.0, 1.0, n)  # 0 at the start of the crossfade, 1 at the end
    angle = np.pi * t / 2  # sweeps 0 -> pi/2 radians (0 -> 90 degrees)
    return np.cos(angle), np.sin(angle)


def equal_power_crossfade(outgoing: np.ndarray, incoming: np.ndarray) -> np.ndarray:
    """Blend two equal-length chunks of audio with an equal-power crossfade."""
    if outgoing.shape != incoming.shape:
        raise ValueError(
            f"chunks must have the same shape, got {outgoing.shape} and {incoming.shape}"
        )
    fade_out, fade_in = equal_power_gains(len(outgoing))
    if outgoing.ndim == 2:
        # One gain per sample, shared by both channels. Reshaping (N,) to (N, 1)
        # lets numpy "broadcast" it across the 2 columns of an (N, 2) array.
        fade_out = fade_out[:, np.newaxis]
        fade_in = fade_in[:, np.newaxis]
    blended = outgoing * fade_out + incoming * fade_in
    return blended.astype(outgoing.dtype, copy=False)


# Stage 4: DJ transitions
# =======================
# Three ways to get from song A to song B, each a function of the same shape as
# equal_power_crossfade (or one built by a helper, when it needs to know more):
#
#     blend      a long beat-matched overlap with a bass swap   bass_swap_blend()
#     drop swap  a hard cut on a downbeat, from A's build into B's drop
#                (just a few milliseconds of equal_power_crossfade, to avoid a click)
#     echo out   A's last beat echoes away, fading through a closing low-pass
#                filter, while B starts                         echo_tail(), echo_join()
#
# Filters: what low-pass and high-pass mean
# -----------------------------------------
# A filter turns some frequencies down and leaves others alone. A LOW-PASS filter
# lets the lows through and cuts everything above its "cutoff" frequency: drums
# lose their snap, cymbals vanish, and the sound gets muffled, as if through a wall.
# A HIGH-PASS filter does the opposite: it cuts the lows and keeps the highs, leaving
# a thin sound with no kick or bass. Use a low-pass at 200 Hz, a high-pass at
# 2.5 kHz, and what's left in between, and you have a 3-band EQ: the "low", "mid"
# and "high" knobs on every DJ mixer.
#
# The filters here are Butterworth filters (as flat as possible in the band they
# keep), run with scipy's sosfiltfilt: forwards and then backwards over the audio,
# which cancels the small time delay a filter normally adds. That matters for the
# EQ: we take the mid band as "everything minus low minus high", so the three bands
# always add back up to exactly the original.
#
# Why two basslines clash
# -----------------------
# Bass notes are long, slow waves (a 50 Hz note is 7 metres long in air), and our
# ears can't separate two of them well. Play two basslines at once and their low
# frequencies pile up: in different keys they beat against each other into a
# wobbling rumble, and even in the same key they double the low-end energy, which
# sounds boomy and muddy and eats up the mix's headroom (that's what made the
# Stage 2 note about kicks adding up). Mids and highs from two songs blend far more
# gracefully. So DJs cut the incoming song's bass for most of a blend, and then, on
# a downbeat, swap: one bass knob down, the other up. One bassline at a time.
#
# The math of an echo
# -------------------
# An echo (a "feedback delay") plays the sound again after a delay D, and feeds that
# repeat back into itself: each repeat is the previous one times the feedback g
# (0 < g < 1). Starting with the last beat of A at level 1, the repeats come at
# D, 2D, 3D ... with levels
#
#     g, g^2, g^3, ...          (a geometric sequence)
#
# so in decibels each repeat is the same step quieter: 20*log10(g) dB (g = 0.55 is
# -5.2 dB per repeat). With D = one beat, A's last beat keeps repeating in time
# while it fades away, and with g < 1 the total, g / (1 - g) of the original, can
# never grow without limit. The closing low-pass filter makes each repeat duller
# too, so A seems to recede into the distance as B takes over.

LOW_CUTOFF = 200.0  # Hz: the low band is everything below this (kick, bass)
HIGH_CUTOFF = 2500.0  # Hz: the high band is everything above this (hats, cymbals, air)
EQ_ORDER = 4  # filter steepness: 4th order cuts 24 dB per octave beyond the cutoff
SWAP_FADE = 0.02  # seconds over which the two bass bands swap, ending on the downbeat
ECHO_FEEDBACK = 0.55  # each echo repeat is this many times the one before
ECHO_SWEEP = (8000.0, 300.0)  # the low-pass cutoff closes from this to this (Hz) over the echo
CLICK_FADE = 0.005  # seconds: tiny fades where audio starts or stops abruptly, to avoid clicks


def three_band(audio: np.ndarray, sr: int, low: float = LOW_CUTOFF, high: float = HIGH_CUTOFF):
    """Split `audio` into (low, mid, high) bands that add up to exactly `audio`."""
    lows = sosfiltfilt(butter(EQ_ORDER, low, "low", fs=sr, output="sos"), audio, axis=0)
    highs = sosfiltfilt(butter(EQ_ORDER, high, "high", fs=sr, output="sos"), audio, axis=0)
    return lows, audio - lows - highs, highs


def bass_swap_gains(n: int, swap_at: int, fade: int) -> tuple[np.ndarray, np.ndarray]:
    """(outgoing, incoming) gains for the low band: A's bass until the swap, then B's.

    The swap is an equal-power crossfade over the `fade` samples ending at `swap_at`,
    so B's bass is fully in, and A's fully out, right on the downbeat.
    """
    fade = max(1, min(fade, swap_at))
    t = np.clip((np.arange(n) - (swap_at - fade)) / fade, 0.0, 1.0)
    return np.cos(np.pi * t / 2), np.sin(np.pi * t / 2)


def bass_swap_blend(outgoing: np.ndarray, incoming: np.ndarray, swap_at: int, sr: int,
                    fade_seconds: float = SWAP_FADE) -> np.ndarray:
    """A blend: mids and highs crossfade (equal power) across the overlap; the bass swaps at `swap_at`.

    `swap_at` is the sample, within the overlap, of the downbeat where B's bass takes
    over. Before it, B plays with its bass cut; after it, A plays with its bass cut.
    """
    if outgoing.shape != incoming.shape:
        raise ValueError(f"chunks must have the same shape, got {outgoing.shape} and {incoming.shape}")
    n = len(outgoing)
    a_low, a_mid, a_high = three_band(outgoing, sr)
    b_low, b_mid, b_high = three_band(incoming, sr)
    fade_out, fade_in = equal_power_gains(n)
    low_out, low_in = bass_swap_gains(n, swap_at, round(fade_seconds * sr))
    if outgoing.ndim == 2:
        fade_out, fade_in, low_out, low_in = (g[:, np.newaxis] for g in (fade_out, fade_in, low_out, low_in))
    blended = a_low * low_out + (a_mid + a_high) * fade_out + b_low * low_in + (b_mid + b_high) * fade_in
    return blended.astype(outgoing.dtype, copy=False)


def bass_swap(swap_at: int, sr: int) -> Transition:
    """A Transition (the usual (outgoing, incoming) shape) that swaps the bass at `swap_at`."""
    return partial(bass_swap_blend, swap_at=swap_at, sr=sr)


def echo_tail(send: np.ndarray, length: int, sr: int, feedback: float = ECHO_FEEDBACK,
              sweep: tuple[float, float] | None = ECHO_SWEEP) -> np.ndarray:
    """The echo of `send` (A's last beat): repeats every len(send) samples, each `feedback` times the last.

    Returns `length` samples. Repeat k (k = 1, 2, ...) starts at (k - 1) * len(send)
    at level feedback**k, so the first one starts right where A stops. With `sweep`,
    a low-pass filter closes from sweep[0] to sweep[1] Hz over the tail. The very
    end fades to silence, so cutting the tail off makes no click.
    """
    delay = len(send)
    tail = np.zeros((length, *send.shape[1:]), dtype=np.float64)
    gain, start = feedback, 0
    while start < length and gain > 1e-4:
        piece = send[: length - start]
        tail[start : start + len(piece)] += gain * piece
        gain *= feedback
        start += delay
    edge = min(length, round(CLICK_FADE * sr))
    ramp = np.linspace(0.0, 1.0, edge)
    if tail.ndim == 2:
        ramp = ramp[:, np.newaxis]
    tail[:edge] *= ramp  # the first repeat starts on a beat: ease it in, no click
    if sweep is not None:
        tail = lowpass_sweep(tail, sr, *sweep)
    tail[length - edge :] *= ramp[::-1]
    return tail.astype(send.dtype, copy=False)


def lowpass_sweep(audio: np.ndarray, sr: int, start_hz: float, end_hz: float, block: int = 512) -> np.ndarray:
    """A low-pass filter whose cutoff glides from start_hz to end_hz (exponentially) over `audio`.

    The filter is redesigned for each short block, carrying its internal state over
    from one block to the next, so the sweep is smooth.
    """
    out = np.empty_like(audio)
    blocks = max(1, -(-len(audio) // block))
    state = None
    for k in range(blocks):
        cutoff = start_hz * (end_hz / start_hz) ** (k / max(1, blocks - 1))  # equal steps in pitch
        sos = butter(2, min(cutoff, 0.45 * sr), "low", fs=sr, output="sos")
        chunk = audio[k * block : (k + 1) * block]
        if state is None:
            state = np.zeros((sos.shape[0], 2, *audio.shape[1:]))
        out[k * block : (k + 1) * block], state = sosfilt(sos, chunk, axis=0, zi=state)
    return out


def echo_join(outgoing: np.ndarray, incoming: np.ndarray, sr: int = 44_100) -> np.ndarray:
    """The echo-out overlap: A's echo tail (already fading on its own) plus B, faded in over a few ms."""
    if outgoing.shape != incoming.shape:
        raise ValueError(f"chunks must have the same shape, got {outgoing.shape} and {incoming.shape}")
    gain = np.minimum(1.0, np.arange(len(incoming)) / max(1, round(CLICK_FADE * sr)))
    if incoming.ndim == 2:
        gain = gain[:, np.newaxis]
    return (outgoing + incoming * gain).astype(outgoing.dtype, copy=False)
