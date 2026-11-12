"""Per-track clean-up and final-mix safety, as pure numpy functions.

Each function takes an audio array (float32, shape (n_samples, 2)) and returns a
result without touching files or changing its input, which makes them easy to test
and to reuse in later stages.

    trim_silence        cut the quiet bits off the start and end of a track
    normalize_loudness  turn each track up or down to the same average loudness
    prevent_clipping    scale the finished mix down if any peak would distort
    click_sound         a short click, for marking beats (--clicks)

Decibels
--------
Our ears judge loudness by ratios: going from amplitude 0.1 to 0.2 feels like the
same step as going from 0.4 to 0.8. Decibels measure exactly those ratios:

    dB = 20 * log10(amplitude)          amplitude = 10 ** (dB / 20)

measured against "full scale" (amplitude 1.0 = 0 dBFS). So:

    1.0 -> 0 dB      0.5 -> -6 dB      0.1 -> -20 dB      0.01 -> -40 dB

Why 20 and not 10? Decibels were defined for POWER ratios as 10 * log10(power),
and power is amplitude squared: 10 * log10(a**2) = 20 * log10(a).
"""

import math

import numpy as np

# Default settings, all in dB.
TARGET_RMS_DB = -16.0  # every track is normalized to this average loudness
SILENCE_THRESHOLD_DB = -40.0  # quieter than this, relative to a track's loudest moment, is "silence"
CEILING_DB = -1.0  # the finished mix never peaks above this

FRAME_LENGTH = 2048  # samples per block when looking for silence (~46 ms at 44.1 kHz)


def db_to_gain(db: float) -> float:
    """Decibels -> amplitude multiplier. -6 dB -> ~0.5, 0 dB -> 1.0."""
    return 10.0 ** (db / 20.0)


def gain_to_db(gain: float) -> float:
    """Amplitude multiplier -> decibels. 0.5 -> ~-6 dB. Silence (0) is -infinity dB."""
    return 20.0 * math.log10(gain) if gain > 0 else -math.inf


def rms(audio: np.ndarray) -> float:
    """Root-mean-square level: one number for how loud `audio` is on average.

    Read the name backwards to get the recipe:

        Square every sample -> negatives become positive, and a squared sample is
                               proportional to the power (energy) it carries
        take the Mean       -> the average power over the whole track
        take the Root       -> back to amplitude units, comparable to the samples

    Why not just average the samples? A waveform swings above and below zero
    equally, so the plain average comes out ~0 for every song. Why not use the peak
    (the single loudest sample)? One loud drum hit would make a quiet song look
    loud. RMS measures energy over time, which tracks how loud music *sounds* far
    better.

    Handy fact, used in the tests: a sine wave of amplitude A has RMS A / sqrt(2),
    about 0.707 * A.
    """
    if audio.size == 0:
        return 0.0
    # Add up in float64: float32 loses precision when summing millions of numbers.
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def rms_db(audio: np.ndarray) -> float:
    """RMS level in dBFS (-infinity for pure silence)."""
    return gain_to_db(rms(audio))


def peak(audio: np.ndarray) -> float:
    """The largest distance from zero of any sample (0.0 for empty audio)."""
    if audio.size == 0:
        return 0.0
    # max/min instead of np.abs(audio).max(): same answer, no full-size temporary array.
    return float(max(audio.max(), -audio.min()))


def trim_silence(
    audio: np.ndarray,
    threshold_db: float = SILENCE_THRESHOLD_DB,
    frame_length: int = FRAME_LENGTH,
) -> np.ndarray:
    """Remove silence (and near-silence) from the start and end of `audio`.

    The track is chopped into short blocks ("frames") and each block's RMS is
    measured. Blocks more than `threshold_db` below the loudest block count as
    silent, and we keep everything from the first non-silent block to the last.
    Quiet passages in the *middle* stay: a breakdown isn't dead air.

    Why compare against the track's loudest block instead of a fixed level? A quiet
    acoustic recording and a loud club master sit at very different absolute levels;
    measuring relative to each track's own peak makes "-40 dB" mean the same thing
    for both. -40 dB is 1/100 of the amplitude, barely audible next to the music,
    so it catches tape hiss and the tails of long fade-outs as well as digital
    silence.

    Returns an empty array if the whole track is silent.
    """
    start, end = non_silent_bounds(audio, threshold_db, frame_length)
    return audio[start:end]


def non_silent_bounds(
    audio: np.ndarray,
    threshold_db: float = SILENCE_THRESHOLD_DB,
    frame_length: int = FRAME_LENGTH,
) -> tuple[int, int]:
    """Where the sound starts and ends: trim_silence() keeps audio[start:end].

    Returns (0, 0) if the whole track is silent. The pipeline needs `start` to shift
    beat times, which are measured from the start of the untrimmed file.
    """
    n = len(audio)
    if n == 0:
        return 0, 0

    # Pad with zeros to a whole number of frames, then reshape so each row holds
    # one frame (with both channels of that frame side by side in the same row).
    pad = -n % frame_length
    padded = np.pad(audio, [(0, pad)] + [(0, 0)] * (audio.ndim - 1))
    frames = padded.reshape(len(padded) // frame_length, -1)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))

    loudest = frame_rms.max()
    if loudest == 0:
        return 0, 0  # digital silence from start to finish

    # Convert the dB threshold into a ratio: -40 dB -> 0.01 x the loudest frame.
    loud_frames = np.flatnonzero(frame_rms > loudest * db_to_gain(threshold_db))
    start = int(loud_frames[0]) * frame_length
    end = min(n, (int(loud_frames[-1]) + 1) * frame_length)
    return start, end


def normalize_loudness(audio: np.ndarray, target_db: float = TARGET_RMS_DB) -> np.ndarray:
    """Scale `audio` so its RMS level is `target_db` dBFS.

    Changing volume is just multiplication: every sample times the same "gain".
    Going from the current RMS to the target RMS takes gain = target / current.
    rms() pools both channels, so they get the same gain and the stereo balance
    doesn't shift.

    Run this AFTER trim_silence: silence is a run of zeros, which drags the average
    down and would make us turn the track up too much.

    For very dynamic tracks (a big gap between average and peak), turning them up
    can push peaks above 1.0. That's fine at this stage: float32 holds values above
    1.0 without damage, and prevent_clipping() deals with it on the finished mix.

    Why -16 dBFS? It's close to the level streaming services normalize to, quiet
    enough that most tracks keep their peaks under the -1 dB ceiling, and loud
    enough that you don't have to crank the volume.

    A limitation: RMS weighs all frequencies equally, but our ears are most
    sensitive around 2-5 kHz, so a bass-heavy track can measure loud yet sound
    quieter. The broadcast standard, LUFS, filters the audio to hear it the way our
    ears do. That's a possible upgrade later.
    """
    current = rms(audio)
    if current == 0.0:
        return audio  # pure silence: no gain can make it louder
    return audio * (db_to_gain(target_db) / current)


def clipping_gain(audio: np.ndarray, ceiling_db: float = CEILING_DB) -> float:
    """The gain (<= 1.0) that brings the loudest peak of `audio` down to `ceiling_db`.

    Returns 1.0, meaning no change, if the audio already stays below the ceiling.
    See prevent_clipping() for why we scale like this.
    """
    loudest = peak(audio)
    ceiling = db_to_gain(ceiling_db)
    return ceiling / loudest if loudest > ceiling else 1.0


def prevent_clipping(audio: np.ndarray, ceiling_db: float = CEILING_DB) -> np.ndarray:
    """Scale the whole mix down, if needed, so no sample peaks above `ceiling_db`.

    Digital audio can't go past +/-1.0: anything beyond gets chopped flat
    ("clipped"), squaring off the tops of the waveform, and you hear that as a harsh
    crackle. Peaks over 1.0 can happen here because normalization turns quiet
    tracks up, and because two tracks add together during a crossfade.

    Why scale everything instead of clipping the few samples that are too loud?
    Scaling keeps the waveform's exact shape, so it adds no distortion. The
    trade-off: one stray peak makes the entire mix a little quieter. A mastering
    "limiter", which smoothly turns down only the loud moments, would avoid that,
    and is a possible upgrade later.

    Why a ceiling of -1 dB (~0.89) and not 0 dB (1.0)? MP3 encoding, and the
    smoothing inside a player's digital-to-analog converter, can create peaks
    slightly higher than any stored sample ("inter-sample peaks"). The extra 1 dB
    of headroom absorbs them.
    """
    gain = clipping_gain(audio, ceiling_db)
    return audio if gain == 1.0 else audio * gain


def click_sound(sr: int, freq: float = 1500.0, level_db: float = -6.0) -> np.ndarray:
    """A short, bright click (mono float32): a sine tone that dies away within 20 ms.

    It starts at full level and decays exponentially (gone to 1% in ~20 ms), so it
    marks a precise moment without blurring into the music.
    """
    t = np.arange(round(0.02 * sr)) / sr
    envelope = np.exp(-t / 0.004)  # e^(-t / 4 ms): 1% left after ~18 ms
    return (db_to_gain(level_db) * envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)
