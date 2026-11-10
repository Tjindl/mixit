"""Track analysis: a placeholder for later stages. Nothing here is used yet.

Each function will take one prepared track (float32, shape (n_samples, 2), at
SAMPLE_RATE) and return one fact about it. The mixer will use those facts to plan
better transitions: automix.load_tracks() is where they'll be computed, and
mixer.plan_crossfades() is what they'll replace.

Roadmap
-------
Stage 2  BPM + beats -> beat matching
         Time-stretch each incoming track to the outgoing track's tempo, and start
         every crossfade on a downbeat so the drums line up. Crossfade lengths
         become musical ("16 bars") instead of a number of seconds.
Stage 3  Key -> harmonic mixing
         Order tracks so neighbours are in compatible keys (the "Camelot wheel").
Stage 4  Energy -> set flow
         Shape the order of the set (warm up, peak, cool down) and pick a transition
         style for each pair of tracks (new functions in transitions.py).
"""

import numpy as np

from audio_io import SAMPLE_RATE


def estimate_bpm(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Tempo in beats per minute.

    TODO (Stage 2): librosa.beat.beat_track() finds the moments where new sounds
    start (an "onset strength" curve) and looks for the steady pulse that best fits
    them. Watch out for half/double-tempo mistakes (64 vs 128 BPM).
    """
    raise NotImplementedError("BPM detection is planned for Stage 2")


def detect_beats(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Time, in seconds, of every beat in the track.

    TODO (Stage 2): librosa.beat.beat_track(..., units="time"). For phrase-aligned
    mixing we'll also want downbeats (the "1" of each bar), which takes more work.
    """
    raise NotImplementedError("beat detection is planned for Stage 2")


def estimate_key(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """Musical key in Camelot notation, e.g. "8A" for A minor.

    TODO (Stage 3): average a chromagram (how much of each of the 12 pitch classes
    is present; librosa.feature.chroma_cqt), then compare it with the
    Krumhansl-Schmuckler profiles of all 24 major and minor keys and pick the
    closest match.
    """
    raise NotImplementedError("key detection is planned for Stage 3")


def estimate_energy(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """One number for how intense the track feels.

    TODO (Stage 4): combine loudness (RMS), brightness (spectral centroid) and
    rhythmic density (onsets per second).
    """
    raise NotImplementedError("energy analysis is planned for Stage 4")
