"""Assemble a list of prepared tracks into one continuous mix.

Building a mix takes two steps:

1. PLAN    decide how many samples each pair of neighbouring tracks overlaps
           -> plan_crossfades()
2. RENDER  lay the tracks end to end and blend every overlap with a transition
           -> render_mix()

mix_tracks() runs both, then makes sure the result can't clip.

Keeping PLAN and RENDER apart is what lets later stages plug in. Stage 2 (beat
matching) needs a smarter plan: crossfades measured in bars and starting on a
downbeat, with tracks cut at cue points and time-stretched to a shared tempo before
rendering. render_mix() itself doesn't have to change.

How the tracks are laid out (3 tracks, 2 crossfades):

    A    |======= A solo =======|-fade out-|
    B                           |-fade in--|===== B solo =====|-fade out-|
    C                                                         |-fade in--|=== C solo ===|

    mix  |======= A solo =======|  A + B   |===== B solo =====|  B + C   |=== C solo ===|
         ^ A starts             ^ B starts                    ^ C starts

Every sample of the mix belongs to exactly one section: either a "solo" section
where one track plays, or a transition where two overlap. So:

    length of mix = sum of track lengths - sum of crossfade lengths
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from processing import CEILING_DB, clipping_gain, gain_to_db
from transitions import Transition, equal_power_crossfade


@dataclass(frozen=True)
class Mix:
    """A finished mix, plus where everything landed. Positions are in samples."""

    audio: np.ndarray  # float32, shape (n_samples, 2)
    starts: list[int]  # where each track starts (begins fading in)
    crossfades: list[int]  # actual overlap of track i and i+1 (may be less than requested)
    safety_gain_db: float = 0.0  # gain applied to avoid clipping: 0 = untouched, -3 = 3 dB quieter


def plan_crossfades(lengths: Sequence[int], crossfade: int) -> list[int]:
    """Return the overlap, in samples, between each pair of neighbouring tracks.

    Normally that's just `crossfade` every time, but a crossfade may use at most
    half of either track it touches. Why half? Track B in the diagram above has a
    fade-in AND a fade-out. If one crossfade could take more than half of B, the
    two fades would overlap inside B and three tracks would play at once. With the
    half rule, fade-in + fade-out <= length of B always holds: a short track's solo
    section can shrink to nothing, but never below zero. That's how a too-short
    song shrinks the crossfade instead of crashing the program.

    TODO (Stage 2): a beat-aware planner replaces this one.
    """
    if crossfade < 0:
        raise ValueError("crossfade can't be negative")
    return [min(crossfade, a // 2, b // 2) for a, b in zip(lengths, lengths[1:])]


def render_mix(
    tracks: Sequence[np.ndarray],
    crossfades: Sequence[int],
    transition: Transition = equal_power_crossfade,
) -> tuple[np.ndarray, list[int]]:
    """Lay `tracks` end to end, overlapping track i and i+1 by `crossfades[i]` samples.

    Returns the mixed audio and the sample where each track starts. Nothing here
    limits the volume, so the result may contain peaks above 1.0.
    """
    if not tracks:
        raise ValueError("need at least one track to mix")
    if len(crossfades) != len(tracks) - 1:
        raise ValueError(f"{len(tracks)} tracks need {len(tracks) - 1} crossfades, got {len(crossfades)}")

    lengths = [len(track) for track in tracks]
    fade_ins = [0, *crossfades]  # track i overlaps the track before it by fade_ins[i]...
    fade_outs = [*crossfades, 0]  # ...and the track after it by fade_outs[i]
    for i, (length, fade_in, fade_out) in enumerate(zip(lengths, fade_ins, fade_outs)):
        if fade_in < 0 or fade_out < 0 or fade_in + fade_out > length:
            raise ValueError(
                f"track {i} is {length} samples long: too short for a {fade_in}-sample "
                f"fade-in plus a {fade_out}-sample fade-out (plan_crossfades prevents this)"
            )

    # Each track starts where the previous one started, plus that track's length,
    # minus the overlap between them.
    starts = [0]
    for length, overlap in zip(lengths, crossfades):
        starts.append(starts[-1] + length - overlap)
    total = starts[-1] + lengths[-1]  # = sum(lengths) - sum(crossfades)

    # Allocate the whole mix once and fill it section by section; that's much
    # faster than growing an array with repeated np.concatenate calls.
    out = np.zeros((total, *tracks[0].shape[1:]), dtype=np.float32)
    for i, track in enumerate(tracks):
        start, length = starts[i], lengths[i]
        fade_in, fade_out = fade_ins[i], fade_outs[i]

        # Solo section: only this track is playing, so copy it straight across.
        out[start + fade_in : start + length - fade_out] = track[fade_in : length - fade_out]

        # Transition: this track's last `fade_out` samples blended with the next
        # track's first `fade_out` samples, written where the next track starts.
        if fade_out > 0:
            next_start = starts[i + 1]
            out[next_start : next_start + fade_out] = transition(
                track[length - fade_out :], tracks[i + 1][:fade_out]
            )
    return out, starts


def mix_tracks(
    tracks: Sequence[np.ndarray],
    crossfade: int,
    transition: Transition = equal_power_crossfade,
    ceiling_db: float = CEILING_DB,
) -> Mix:
    """Crossfade `tracks` together, in order, into one finished mix.

    `crossfade` is the requested overlap in samples (seconds * sample rate); it's
    shortened automatically wherever a track is too short for it.
    """
    crossfades = plan_crossfades([len(track) for track in tracks], crossfade)
    audio, starts = render_mix(tracks, crossfades, transition)

    # Turn the whole mix down if any peak would clip; this is what
    # processing.prevent_clipping() does, but scaling in place (*=) skips making a
    # second full-length copy of the mix in memory. That's safe because render_mix
    # just created this buffer and nothing else uses it.
    gain = clipping_gain(audio, ceiling_db)
    if gain < 1.0:
        audio *= gain
    return Mix(audio, starts, crossfades, safety_gain_db=gain_to_db(gain))
