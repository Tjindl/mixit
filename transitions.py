"""Transitions: how one track hands over to the next.

A transition is any function with this shape:

    blended = transition(outgoing, incoming)

`outgoing` is the last N samples of the track that's ending, `incoming` is the
first N samples of the track that's starting (both shape (N, 2)), and `blended` is
the N samples the listener hears while both are playing.

Stage 1 has one transition, the equal-power crossfade. Later stages can add others
with the same shape (a "bass swap" that EQs out one track's low end, a filter
sweep, an echo-out) and hand them to the mixer without changing the mixer.
"""

from collections.abc import Callable

import numpy as np

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
