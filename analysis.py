"""Song analysis: tempo and beats (Stage 2), key and energy (Stage 3).

detect_beats() finds a song's tempo (BPM) and the time of every beat, in four steps.
librosa provides the core of each; we sharpen steps 2 and 4, because beat matching
needs a tempo accurate to a few hundredths of a BPM:

    1. onset strength   how much new sound starts at each moment
    2. tempo            the steady pulse that best explains those onsets
    3. beat positions   exactly where each beat falls (dynamic programming)
    4. final BPM        a straight line fitted through all the beats

octave_factors() then fixes the classic beat-tracker mistake of reading a song at
half or double its real tempo.

estimate_key() finds the musical key ("A minor", Camelot "8A"), and
energy_features() / energy_scores() rate how intense each song is; ordering.py
uses both to decide which song should follow which. AnalysisCache saves all of it
in a JSON file so each song is only analyzed once.
"""

import importlib.util
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import librosa
import numpy as np

from audio_io import SAMPLE_RATE
from processing import non_silent_bounds, rms_db

HOP_LENGTH = 512  # onset curve resolution: one value per 512 samples (11.6 ms at 44.1 kHz)
MIN_BEATS = 8  # fewer beats than this means no steady beat (ambient, speech, silence)
TIGHTNESS = 1600  # how strictly beats must keep to the tempo (librosa's default is 100; see step 3)

# Is there a beat at all? librosa rescales the onset curve before tracking, so it
# will "find" a pulse in anything, even the tiny ripples of a steady tone. Two checks
# on the onset strength AT the detected beats reject that (values measured on test
# tones, noise, drum loops and real songs):
MIN_BEAT_STRENGTH = 0.75  # dB of rise, averaged over bands: drums give 1.5-35, held notes under 0.4
MIN_BEAT_CONTRAST = 1.4  # times the song's average onset strength: real songs gave 3.5+,
#                          a soft kick under steady noise 1.7, pure noise only ~1.2.
# Both lean toward "yes, there's a beat": wrongly matching a beatless song is
# harmless, but wrongly giving up on a real beat loses the beat matching.

# Step 1: onset strength (spectral flux)
# --------------------------------------
# A beat is felt where new sound starts: a drum hit, a chord change. To find those
# moments, librosa cuts the song into short overlapping windows (46 ms long, one
# every 11.6 ms) and measures the energy in each of 128 frequency bands (a "mel
# spectrogram", in dB). Then, for every band, it asks: did the energy go UP since
# the previous window? Adding up only the increases gives one number per window,
# the onset strength, also called spectral flux:
#
#     flux[t] = average over bands f of  max(0, S[f, t] - S[f, t - 1])
#
# It spikes when something new starts, and stays near zero while notes are held,
# because a sustained note doesn't get any louder. Decreases are ignored because a
# sound dying away isn't a new event.
#
# Step 2: tempo (autocorrelation and the tempogram)
# -------------------------------------------------
# For a song with a steady beat, the onset curve is roughly periodic: a spike every
# beat. Autocorrelation measures how well a curve lines up with a delayed copy of
# itself:
#
#     ac[lag] = sum over t of  flux[t] * flux[t + lag]
#
# If beats come every 0.47 s, shifting the curve by 0.47 s lays each spike on top
# of the next one, so ac[0.47 s] is large. It's also large at 0.94 s, 1.41 s, ...
# (every multiple of the beat period). librosa computes this over sliding 8-second
# windows (that grid of autocorrelations over time is the "tempogram"), averages
# them, converts each lag to a tempo with BPM = 60 / lag_in_seconds, weights them by
# a prior that favours tempos near 120 BPM, and picks the strongest.
#
# librosa picks a lag in whole frames, though, so its answer is only accurate to
# about +/-1.5 BPM (it reads a 128 BPM click track as 129.2, and one of the test
# songs as 132.5 when it's really 134). For beat matching that's far too coarse: a
# 1 BPM error drifts 60 ms over a 16-beat crossfade. So we keep librosa's choice of
# octave (128 rather than 64), and sharpen it ourselves (see _refine_tempo): take
# the autocorrelation of the WHOLE song's onset curve, find its peak within 10% of
# librosa's tempo, and pinpoint the top of that peak between frames.
#
# Step 3: beat positions (dynamic programming)
# --------------------------------------------
# Knowing the tempo isn't enough: we need WHERE each beat falls. The tracker wants
# beat times that (a) land on strong onsets, and (b) are spaced about one period
# apart. It scores a whole sequence of beats as
#
#     sum of onset strength at the beats  -  tightness * sum of log(gap / period)^2
#
# There are astronomically many possible sequences, but dynamic programming finds
# the best one in a single pass. Going through the song frame by frame, it works out
# best[t], the score of the best sequence whose last beat is at frame t:
#
#     best[t] = flux[t] + max over earlier frames p of (best[p] - tightness * log((t - p) / period)^2)
#
# and remembers which p won. The sequence ending at frame t must have SOME previous
# beat p, and the best way to reach p is already known; that reuse is the whole
# trick. At the end it starts from the best final beat and follows the remembered
# "previous beat" links backwards to recover every beat.
#
# We run it at our sharpened tempo, with tightness 1600 instead of librosa's 100.
# With a loose setting, beats wander off the grid to grab strong off-beat sounds
# (syncopated synths, vocals); on one test song that made a steady 125.4 BPM track
# measure as 126.6. Dance music sits on a rigid grid, so we hold the tracker to it.
#
# Step 4: final BPM (a line through the beats)
# --------------------------------------------
# Fit a straight line through the beats: beat time against beat number, by least
# squares. Its slope is the time per beat. On a steady track it's accurate to a few
# hundredths of a BPM, sharper still than step 2, which is limited by the frame grid
# on very sharp sounds like click tracks. If the line disagrees with step 2 by more
# than 1%, the beats are too irregular to trust and we keep step 2's tempo.

# Half/double tempo correction (see octave_factors)
OCTAVE_TOLERANCE = 0.2  # a halved/doubled reading must land within 20% of the consensus
TYPICAL_BPM = 120.0  # breaks ties: prefer readings nearer this


@dataclass(frozen=True)
class BeatAnalysis:
    """What beat tracking found in one song."""

    bpm: float | None  # tempo in beats per minute; None if no steady beat was found
    beats: np.ndarray  # beat times in seconds, from the start of the analyzed audio


def detect_beats(audio: np.ndarray, sr: int = SAMPLE_RATE) -> BeatAnalysis:
    """Find the tempo and beat times of `audio` (float32, shape (n,) or (n, 2))."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio  # beat tracking needs one channel
    onset_env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=HOP_LENGTH)  # step 1

    # Step 2: librosa's estimate, then sharpened. Newer librosa returns the tempo as a
    # 1-element numpy array (older versions as a plain number); np.atleast_1d(...)[0]
    # handles both. beat_track() returns its tempo the same way.
    librosa_bpm = float(np.atleast_1d(
        librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
    )[0])
    if librosa_bpm <= 0 or not onset_env.any():
        return BeatAnalysis(None, np.zeros(0))  # silence: nothing to track
    tempo = _refine_tempo(onset_env, sr, librosa_bpm)
    tempo = _check_three_quarters(onset_env, sr, tempo)

    _, frames = librosa.beat.beat_track(  # step 3, at our tempo
        onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH,
        bpm=tempo, tightness=TIGHTNESS, trim=False,
    )
    frames = _trim_weak_beats(onset_env, frames)
    # The onset curve reacts about one frame after a sound starts (it measures the
    # change from one window to the next), so beat times come out ~11 ms late.
    # Shift them back by one frame; you'd hear the difference with --clicks.
    exact_frames = _snap_to_onset_peaks(onset_env, frames)
    beats = np.maximum((exact_frames - 1) * HOP_LENGTH / sr, 0.0)

    if len(beats) < MIN_BEATS:
        return BeatAnalysis(None, beats)
    at_beats = float(onset_env[frames].mean())
    if at_beats < MIN_BEAT_STRENGTH or at_beats < MIN_BEAT_CONTRAST * float(onset_env.mean()):
        # Either nothing much starts at the "beats" (a held chord, a pad), or onsets
        # are everywhere and the beats don't stand out (noise): no steady beat.
        return BeatAnalysis(None, beats)

    _, period, _ = fit_beat_grid(beats)  # step 4
    line_bpm = 60.0 / period
    bpm = line_bpm if abs(line_bpm / tempo - 1) <= 0.01 else tempo  # irregular beats: keep step 2
    return BeatAnalysis(bpm, beats)


def _refine_tempo(onset_env: np.ndarray, sr: int, rough_bpm: float) -> float:
    """Sharpen a tempo estimate using the autocorrelation peak of the whole onset curve.

    Computing the autocorrelation at every lag directly would take ages, so we use
    the FFT: by the Wiener-Khinchin theorem, a signal's autocorrelation is the inverse
    FFT of its power spectrum, |FFT|^2. (Padding to twice the length stops the end of
    the song from wrapping around onto its start.) Then we look for the highest
    peak within 10% of `rough_bpm`, and pinpoint its top between frames by fitting a
    parabola through the three values around it.
    """
    frames_per_second = sr / HOP_LENGTH
    autocorr = _autocorrelation(onset_env)

    lag_min = max(2, int(frames_per_second * 60 / (rough_bpm * 1.1)))  # lags are in frames
    lag_max = min(len(autocorr) - 2, int(np.ceil(frames_per_second * 60 / (rough_bpm * 0.9))))
    if lag_max <= lag_min:
        return rough_bpm
    lag = lag_min + int(np.argmax(autocorr[lag_min : lag_max + 1]))
    y0, y1, y2 = autocorr[lag - 1], autocorr[lag], autocorr[lag + 1]
    curvature = y0 - 2 * y1 + y2
    exact_lag = lag + (0.5 * (y0 - y2) / curvature if curvature < 0 else 0.0)
    return 60.0 * frames_per_second / exact_lag


def _autocorrelation(onset_env: np.ndarray) -> np.ndarray:
    """The onset curve's autocorrelation at every lag (in frames), via the FFT (see _refine_tempo)."""
    curve = onset_env - onset_env.mean()
    spectrum = np.fft.rfft(curve, 2 * len(curve))
    return np.fft.irfft(np.abs(spectrum) ** 2)[: len(curve)]


# The 3:4 trap: tresillo rhythms
# ------------------------------
# A lot of pop, reggaeton, dancehall and Punjabi music is built on the "tresillo"
# (or dembow) pattern: every 2 beats split into 3 + 3 + 2 sixteenth notes, with a
# drum or bass hit at the start of each group. Hits 3 sixteenths apart come every
# 3/4 of a beat, which is exactly what a beat at 4/3 the tempo would sound like, and
# beat trackers fall for it: a 95 BPM song reads as 126.7. librosa's tempo estimate
# is especially prone to it, because it judges from short windows and leans toward
# 120 BPM, which the wrong answer is closer to.
#
# The whole song usually knows better: summed over minutes, the real beat repeats
# more consistently than the tresillo, so its autocorrelation peak is higher. (On a
# real 95 BPM song read as 126.5, the peak at 95 was 1.21 times the one at 126.5.)
# So we compare the two, and take the slower tempo if its peak is clearly higher.
# We don't check the other way round (a 4/3 faster tempo): songs with busy
# sixteenth-note syncopation have strong peaks there, and switching would break them.

TRESILLO_MARGIN = 1.1  # switch to 3/4 of the tempo only if its pulse is this much stronger


def _check_three_quarters(onset_env: np.ndarray, sr: int, tempo: float) -> float:
    """3/4 of `tempo` if the whole song's pulse is clearly stronger there (see above), else `tempo`."""
    autocorr = _autocorrelation(onset_env)
    if autocorr[0] <= 0:
        return tempo
    frames_per_second = sr / HOP_LENGTH

    def pulse(bpm: float) -> float:  # the autocorrelation peak within 2% of this tempo's lag
        lag = frames_per_second * 60 / bpm
        lo, hi = max(1, int(lag * 0.98)), min(len(autocorr) - 1, int(np.ceil(lag * 1.02)))
        return float(autocorr[lo : hi + 1].max()) / autocorr[0] if hi >= lo else 0.0

    slower = 0.75 * tempo
    if pulse(slower) > TRESILLO_MARGIN * max(pulse(tempo), 1e-9):
        return _refine_tempo(onset_env, sr, slower)
    return tempo


def _trim_weak_beats(onset_env: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Drop leading and trailing beats that have no real onset under them.

    Left to itself, the tracker keeps placing beats before the music starts and
    after it stops. librosa can trim those (trim=True), but in testing it also threw
    away the last 4 seconds of genuine beats from one track, exactly where we mix
    out. So we trim ourselves: strip beats from each end while the onset strength
    near the beat is under a quarter of its typical value.
    """
    if len(frames) == 0:
        return frames
    # The strongest onset within 2 frames of each beat: beats can sit a frame or two off the peak.
    starts = np.clip(frames - 2, 0, len(onset_env) - 1)
    ends = np.clip(frames + 3, 1, len(onset_env))
    strength = np.array([onset_env[a:b].max() for a, b in zip(starts, ends)])
    strong = strength >= 0.25 * np.median(strength)
    first, last = int(np.argmax(strong)), len(strong) - int(np.argmax(strong[::-1]))
    return frames[first:last]


def _snap_to_onset_peaks(onset_env: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Move each beat onto the exact peak of the onset curve beside it, between frames.

    The tracker can only place beats on whole frames (11.6 ms apart), and it may sit
    a frame or two to one side of the actual hit. For each beat we find the onset
    peak within 2 frames and estimate where the true peak lies between frames by
    fitting a parabola through the three values around it: for heights y0, y1, y2
    at frames p-1, p, p+1, the vertex is at
        p + (y0 - y2) / (2 * (y0 - 2*y1 + y2))
    Returns fractional frame numbers.
    """
    exact = []
    for frame in frames:
        a, b = max(0, frame - 2), min(len(onset_env), frame + 3)
        peak = a + int(np.argmax(onset_env[a:b]))
        offset = 0.0
        if 0 < peak < len(onset_env) - 1:
            y0, y1, y2 = onset_env[peak - 1], onset_env[peak], onset_env[peak + 1]
            curvature = y0 - 2 * y1 + y2
            if curvature < 0:  # a real peak (curving downwards)
                offset = float(np.clip(0.5 * (y0 - y2) / curvature, -0.5, 0.5))
        exact.append(peak + offset)
    return np.array(exact, dtype=float)


def fit_beat_grid(beats: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Fit evenly spaced beats, `first + k * period`, through detected beat times.

    Returns (first, period, numbers): the fitted time of beat 0, the time between
    beats, and the beat number k given to each detected beat. Numbering comes from
    the gaps, so a beat the tracker missed (a gap of 2 periods) bumps the count by 2
    instead of throwing the line off.

    The fit runs twice. The tracker sometimes misplaces a beat badly (most often the
    very last one, near the end of the file), and least squares lets one bad point
    tilt the whole line. So after the first fit we drop beats more than 1/8 of a beat
    away from the line and fit again, as long as most beats survive. If many don't,
    the tempo is drifting and the first fit is the best we can do.
    """
    if len(beats) < 2:
        raise ValueError("need at least 2 beats to fit a beat grid")
    gaps = np.diff(beats)
    steps = np.maximum(1, np.round(gaps / np.median(gaps)))  # 1 normally, 2 after a missed beat
    numbers = np.concatenate([[0.0], np.cumsum(steps)])
    period, first = np.polyfit(numbers, beats, 1)  # least-squares line: slope, intercept

    keep = np.abs(beats - (first + numbers * period)) < period / 8
    if not keep.all() and keep.sum() >= max(2, 0.75 * len(beats)):
        period, first = np.polyfit(numbers[keep], beats[keep], 1)
    return float(first), float(period), numbers


# Half/double tempo errors
# ------------------------
# In step 2, a 128 BPM song's onset curve also repeats every 2 beats (64 BPM), and
# its hi-hats may tick every half beat (256 BPM). All of those lags score well, and
# the 120 BPM prior usually, but not always, picks the right one. Comparing the
# songs in a set exposes the mistake: a "64 BPM" song among 128s is almost certainly
# a 128 that was read at half speed.


def octave_factors(bpms: list[float | None], reference: float | None = None) -> list[float]:
    """For each tempo reading, the factor (0.5, 1 or 2) that fixes a half/double error.

    A reading gets doubled (or halved) only if that lands it within
    OCTAVE_TOLERANCE of the reference tempo. Without a `reference` (from --bpm), the
    reference is the consensus of the set. We try each song's tempo as the reference
    and keep the one that the most songs agree with (as they are, or halved or
    doubled), then the one needing the fewest corrections, then the one nearest
    120 BPM (which settles one-against-one ties like a 64 and a 128). Songs without
    a tempo (None) get 1.
    """
    readings = [bpm for bpm in bpms if bpm]
    if not readings:
        return [1.0] * len(bpms)

    def best_factor(bpm: float, ref: float) -> float | None:
        """0.5, 1 or 2, whichever brings `bpm` within tolerance of `ref`; None if none does."""
        factor = min((0.5, 1.0, 2.0), key=lambda f: abs(math.log2(bpm * f / ref)))
        close_enough = abs(math.log2(bpm * factor / ref)) <= math.log2(1 + OCTAVE_TOLERANCE)
        return factor if close_enough else None

    if reference is None:
        def score(ref: float) -> tuple[int, int, float]:
            factors = [best_factor(bpm, ref) for bpm in readings]
            agreeing = sum(f is not None for f in factors)
            corrections = sum(f not in (None, 1.0) for f in factors)
            return -agreeing, corrections, abs(math.log2(ref / TYPICAL_BPM))

        reference = min(readings, key=score)
    return [(best_factor(bpm, reference) or 1.0) if bpm else 1.0 for bpm in bpms]


def apply_octave(analysis: BeatAnalysis, factor: float) -> BeatAnalysis:
    """`analysis` with its tempo multiplied by `factor` (2 or 0.5), beats to match."""
    if factor == 1.0 or analysis.bpm is None:
        return analysis
    return replace(analysis, bpm=analysis.bpm * factor, beats=octave_beats(analysis.beats, factor))


def octave_beats(beats: np.ndarray, factor: float) -> np.ndarray:
    """Beat times for the same song counted at `factor` (0.5, 1 or 2) times the tempo."""
    if factor == 1.0:
        return beats
    if factor == 2.0:
        # The tracker found every other beat: add the ones halfway in between.
        return np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2]))
    if factor == 0.5:
        # It found two "beats" per real beat: keep every other one. Which half we
        # keep doesn't matter for keeping beats in sync; it would matter for lining
        # up bars (downbeats), which is a later stage.
        return beats[::2]
    raise ValueError(f"factor must be 0.5, 1 or 2, got {factor}")


# Downbeats: where each bar starts
# --------------------------------
# Beats come in bars of 4 (almost all dance and pop music is in 4/4), and the first
# beat of each bar, the downbeat, is where things happen: chords change, new parts
# come in, a drop hits. Beat tracking finds the beats but not which one is "1".
#
# Our fallback asks the kick drum and the bass. Producers usually put the strongest
# kick, or a new bass note, on beat 1. So we measure low-frequency onset strength
# (spectral flux, as in step 1 above, but only over 30-200 Hz) at every beat, and
# average it separately over the 4 possible phases: beats 0, 4, 8...; beats 1, 5,
# 9...; and so on. The phase with the strongest lows is taken as beat 1.
#
# It's a simple rule, and it fails on songs whose kick is equally loud on every beat
# and whose bass doesn't change on the bar line. The beat_this package (a neural
# network trained on thousands of annotated songs) is far more reliable, so if it's
# installed, its beats and downbeats are used instead. Check the result with
# --cue-debug, which marks every bar in Audacity.

BEATS_PER_BAR = 4
DOWNBEAT_BASS_RANGE = (30.0, 200.0)  # Hz: kick drum and bass


@dataclass(frozen=True)
class Downbeats:
    times: np.ndarray  # downbeat times in seconds
    beats_per_bar: int
    method: str  # "bass" (our fallback) or "beat_this"


def beat_this_available() -> bool:
    return importlib.util.find_spec("beat_this") is not None


def find_downbeats(audio: np.ndarray, beats: np.ndarray, sr: int = SAMPLE_RATE) -> Downbeats:
    """Which of the beats start a bar: the phase with the strongest kick and bass (see above)."""
    if len(beats) < 2 * BEATS_PER_BAR:
        return Downbeats(beats[:0], BEATS_PER_BAR, "bass")
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    y = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=ANALYSIS_SR) if sr != ANALYSIS_SR else mono
    hop = HOP_LENGTH // 2  # 11.6 ms at 22.05 kHz, as for beat tracking
    power = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=ANALYSIS_SR, n_fft=2048)
    low, high = DOWNBEAT_BASS_RANGE
    lows_db = 10 * np.log10(power[(freqs >= low) & (freqs <= high)].sum(axis=0) + 1e-10)
    flux = np.maximum(0.0, np.diff(lows_db, prepend=lows_db[0]))  # how much the lows jump up
    frames = np.clip(np.round(beats * ANALYSIS_SR / hop).astype(int), 0, len(flux) - 1)
    strength = np.array([flux[max(0, f - 2) : f + 3].max() for f in frames])
    _, _, numbers = fit_beat_grid(beats)  # beat numbers, counting any beat the tracker missed
    phase = numbers.astype(int) % BEATS_PER_BAR
    scores = [strength[phase == p].mean() if np.any(phase == p) else -np.inf for p in range(BEATS_PER_BAR)]
    return Downbeats(beats[phase == int(np.argmax(scores))], BEATS_PER_BAR, "bass")


_beat_this_model = None


def beat_this_beats(audio: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, Downbeats] | None:
    """Beats and downbeats from beat_this (None if it fails). Needs PyTorch.

    The first call loads the model, downloading it (~80 MB) the first time ever.
    """
    global _beat_this_model
    try:
        if _beat_this_model is None:
            from beat_this.inference import Audio2Beats

            _beat_this_model = Audio2Beats(checkpoint_path="final0", device="cpu", dbn=False)
        beats, downbeats = _beat_this_model(audio.astype(np.float32), sr)
    except Exception as exc:  # a broken install shouldn't stop the mix
        print(f"  warning: beat_this failed ({exc}); using our own beat tracking", file=sys.stderr)
        return None
    beats, downbeats = np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)
    if len(downbeats) >= 2:
        per_bar = int(round(np.median(np.diff(np.searchsorted(beats, downbeats)))))
    else:
        per_bar = BEATS_PER_BAR
    return beats, Downbeats(downbeats, max(2, per_bar), "beat_this")


# Key detection
# -------------
# What a chroma vector is
#   Western music uses 12 pitch classes, C C# D ... B; every C on the piano, low or
#   high, is the same pitch class (each octave doubles the frequency). A chromagram
#   measures, moment by moment, how much energy falls on each of the 12. librosa's
#   chroma_cqt gets there through a constant-Q transform: a spectrum whose bins are
#   spaced like piano keys (12 per octave) instead of evenly in Hz, so each bin is
#   one note, and folding the octaves together gives 12 numbers. Averaged over the
#   whole song, that's the chroma vector: which notes the song uses, and how much.
#
#   We start the spectrum at C3 (131 Hz), not at librosa's default C1 (33 Hz).
#   Kick drums and 808 basses boom below that, on every beat, often loosely tuned;
#   on test songs with a 50 Hz kick, the default range heard it as a loud "G" and got
#   2 of 5 keys wrong. Above C3 all 5 were right. Bass lines aren't lost: their
#   overtones above C3 carry the same pitch classes.
#
# Comparing it with key profiles
#   Krumhansl and Schmuckler had listeners rate how well each of the 12 notes "fits"
#   after hearing a key established. The ratings (MAJOR_PROFILE, MINOR_PROFILE
#   below, starting from the key's root) are highest on the root, then the 5th and
#   the 3rd, then the rest of the scale, and lowest on notes outside it. A song in a
#   key uses notes in roughly those proportions. To test "is this song in G major?",
#   rotate the major profile so its first value lands on G, and measure how well
#   the song's chroma vector goes up and down with it: the Pearson correlation
#
#       r = sum((x - mean x) * (y - mean y)) / sqrt(sum((x - mean x)^2) * sum((y - mean y)^2))
#
#   which is +1 for a perfect match, 0 for no relation, and ignores overall level.
#   Try all 12 roots x major/minor = 24 candidates; the highest r is the key.
#
# How sure we are
#   Two things make a key detection trustworthy, and the confidence multiplies them:
#
#   1. A clear winner. If the best key barely beats the runner-up, the song is
#      ambiguous. A gap of KEY_FULL_CONFIDENCE_GAP or more counts fully. Measured:
#      chords, scales and test songs in a clear key gave gaps of 0.19 to 0.32, two
#      real songs 0.19 and 0.21. (The runner-up is usually a close relative, like
#      the relative minor, which shares every note; so even clear songs rarely
#      reach 0.3.)
#   2. Notes that stand out at all. Correlation ignores scale, so it finds a
#      "pattern" even in the tiny random ripples of a flat chroma vector: on white
#      noise the gap alone came out as high as 0.18. So we also measure the
#      vector's spread (standard deviation / mean): white noise measured 0.02, real
#      songs 0.3 to 0.4, plain chords 0.6 to 0.9. A spread of
#      KEY_FULL_CONFIDENCE_SPREAD or more counts fully, so noise ends up near 0.

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE_NAMES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
KEY_FULL_CONFIDENCE_GAP = 0.15  # best minus runner-up correlation that counts as fully confident
KEY_FULL_CONFIDENCE_SPREAD = 0.3  # chroma spread (std / mean) that counts as fully tonal
ANALYSIS_SR = 22_050  # key and energy analysis run at half rate: nothing they need is above 11 kHz
CHROMA_HOP = 2048  # one chroma frame per 93 ms at 22.05 kHz: plenty for a whole-song average
CHROMA_LOWEST = "C3"  # chroma ignores everything below this note (kick drums; see above)
CHROMA_OCTAVES = 5  # ...and above 5 octaves higher (C8, 4.2 kHz)


@dataclass(frozen=True)
class KeyEstimate:
    key: str | None  # e.g. "A minor"; None if the song has no pitched content at all
    confidence: float  # 0 (a guess) to 1 (clear)
    method: str  # "essentia" or "librosa"


def key_method() -> str:
    """Which key detector estimate_key() uses: essentia's if it's installed."""
    return "essentia" if importlib.util.find_spec("essentia") is not None else "librosa"


def estimate_key(audio: np.ndarray, sr: int = SAMPLE_RATE) -> KeyEstimate:
    """The song's musical key, e.g. KeyEstimate("A minor", 0.8, "librosa").

    Uses essentia's KeyExtractor when essentia is installed (it's tuned on real
    recordings, so it's usually more accurate), and the chroma + Krumhansl-Schmuckler
    method below otherwise, or if essentia fails.
    """
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    chroma = chroma_vector(mono, sr)
    if not np.any(chroma):  # silence: no notes at all, so no key
        return KeyEstimate(None, 0.0, key_method())
    if key_method() == "essentia":
        try:
            return _essentia_key(mono, sr, chroma)
        except Exception as exc:  # an essentia problem shouldn't stop the mix
            print(f"  warning: essentia's key detection failed ({exc}); using librosa's",
                  file=sys.stderr)
    return key_from_chroma(chroma)


def chroma_vector(mono: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """How much energy falls on each of the 12 pitch classes (C first), averaged over the song."""
    y = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=ANALYSIS_SR) if sr != ANALYSIS_SR else mono
    if not np.any(y):
        return np.zeros(12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # librosa warns about tuning on near-silent input
        chroma = librosa.feature.chroma_cqt(y=y, sr=ANALYSIS_SR, hop_length=CHROMA_HOP,
                                            fmin=librosa.note_to_hz(CHROMA_LOWEST), n_octaves=CHROMA_OCTAVES)
    return chroma.mean(axis=1)


def key_from_chroma(chroma: np.ndarray) -> KeyEstimate:
    """Correlate a chroma vector with all 24 rotated key profiles; the best one is the key."""
    if np.ptp(chroma) < 1e-9:  # flat or empty: no notes stand out, so no key
        return KeyEstimate(None, 0.0, "librosa")
    scores = []
    for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
        for root in range(12):
            # np.roll moves the profile's first value (the root) to position `root`.
            r = np.corrcoef(chroma, np.roll(profile, root))[0, 1]
            scores.append((float(r), f"{NOTE_NAMES[root]} {mode}"))
    scores.sort(reverse=True)
    clear_winner = min(1.0, (scores[0][0] - scores[1][0]) / KEY_FULL_CONFIDENCE_GAP)
    return KeyEstimate(scores[0][1], float(clear_winner * _notes_stand_out(chroma)), "librosa")


def _notes_stand_out(chroma: np.ndarray) -> float:
    """0..1: how uneven the chroma vector is (see "How sure we are"); 0 for a flat one."""
    mean = float(np.mean(chroma))
    return min(1.0, float(np.std(chroma)) / mean / KEY_FULL_CONFIDENCE_SPREAD) if mean > 0 else 0.0


def _essentia_key(mono: np.ndarray, sr: int, chroma: np.ndarray) -> KeyEstimate:
    """essentia's key detector.

    Its "strength" (0..1) is the correlation of the best key, which has the same
    blind spot as ours: on white noise it measured 0.74. So it's scaled by the same
    "notes stand out" check, which brings noise down to ~0.05.
    """
    import essentia.standard as es

    # The "edma" profile was built for electronic dance music.
    key, scale, strength = es.KeyExtractor(sampleRate=float(sr), profileType="edma")(mono.astype(np.float32))
    pitch_class, _ = parse_key(f"{key} {scale}")
    confidence = float(np.clip(strength, 0.0, 1.0)) * _notes_stand_out(chroma)
    return KeyEstimate(f"{NOTE_NAMES[pitch_class]} {scale}", confidence, "essentia")


# The circle of fifths and the Camelot wheel
# ------------------------------------------
# A major scale has 7 of the 12 notes: C major is C D E F G A B. Start a major
# scale on its 5th note instead (G) and you get G A B C D E F#: the same notes except
# one (F becomes F#). Keep stepping up by fifths (C, G, D, A, E, ...) and each key
# differs from the last by exactly one note; after 12 steps you're back at C. That
# loop is the circle of fifths. Neighbours on it share 6 of their 7 notes, so a
# melody or chord from one rarely clashes with the other: that's why mixing into
# a neighbouring key sounds smooth.
#
# Every major key also has a RELATIVE minor with exactly the same 7 notes, starting
# 3 semitones lower: C major and A minor are both just the white keys.
#
# DJs use the Camelot wheel, the circle of fifths with easy labels: numbers 1-12 go
# round the circle (one step = one fifth), "B" is the major key and "A" its relative
# minor at the same number. C major is 8B, A minor 8A, G major 9B, E minor 9A. So:
#
#     same code            same key                        all 7 notes shared
#     +-1 number           a fifth up or down              6 of 7 shared
#     same number, A <-> B relative major/minor            all 7 shared, new mood
#     +2 numbers           a whole tone up ("energy boost") 5 of 7 shared, a lift
#
# Each step round the wheel adds 7 semitones, so the wheel number of a major key
# with root pitch class p (C = 0) is (7p mod 12 + 7) mod 12 + 1; a minor key uses
# its relative major's number.


def parse_key(name: str) -> tuple[int, str]:
    """"F# minor" -> (6, "minor"): pitch class (C = 0) and mode. Sharps or flats ("Gb major")."""
    note, mode = name.strip().split()
    letters = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    pitch = letters[note[0].upper()] + note[1:].count("#") - note[1:].count("b")
    mode = mode.lower()
    if mode not in ("major", "minor"):
        raise ValueError(f"not a key: {name!r}")
    return pitch % 12, mode


def camelot(key: str | None) -> str | None:
    """A key in Camelot notation: "A minor" -> "8A", "C major" -> "8B"."""
    if key is None:
        return None
    pitch, mode = parse_key(key)
    major_root = pitch if mode == "major" else (pitch + 3) % 12  # a minor key's relative major
    number = (7 * major_root % 12 + 7) % 12 + 1
    return f"{number}{'B' if mode == 'major' else 'A'}"


# Energy
# ------
# How intense a song feels, from three measurements, each one a clue:
#
#   loudness    RMS level of the file, in dB. The mixer evens out loudness, so this
#               isn't about volume in the mix: it shows how hard the track was
#               mastered, and dense, compressed, "loud" masters are high-energy music.
#   onset rate  new sounds (drum hits, notes) per second: how busy the rhythm is.
#   brightness  the spectral centroid, the "centre of mass" of the spectrum, in Hz.
#               Hi-hats, cymbals and distorted synths push it up; mellow pads and
#               bass keep it low. Pitch is heard in octaves, so we compare it in
#               octaves (log2) too.
#
# energy_scores() puts each measurement on a 0..1 scale ACROSS THE SET (the
# quietest song 0, the loudest 1) and takes a weighted average. A guard keeps
# near-identical songs near-identical: each scale spans at least ENERGY_MIN_SPREAD,
# so two songs 0.5 dB apart don't come out as 0 and 1.

ENERGY_WEIGHTS = {"loudness": 0.4, "onset_rate": 0.35, "brightness": 0.25}
ENERGY_MIN_SPREAD = {"loudness": 6.0, "onset_rate": 2.0, "brightness": 1.0}  # dB, onsets/s, octaves


@dataclass(frozen=True)
class EnergyFeatures:
    loudness_db: float  # RMS level of the song's non-silent part, dBFS
    onset_rate: float  # onsets per second
    brightness_hz: float  # median spectral centroid


def energy_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> EnergyFeatures | None:
    """Measure a song's loudness, onset rate and brightness (None if it's silent)."""
    start, end = non_silent_bounds(audio)
    if end <= start:
        return None
    audio = audio[start:end]
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    y = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=ANALYSIS_SR) if sr != ANALYSIS_SR else mono
    onsets = librosa.onset.onset_detect(y=y, sr=ANALYSIS_SR, hop_length=HOP_LENGTH // 2)
    centroid = librosa.feature.spectral_centroid(y=y, sr=ANALYSIS_SR, n_fft=4096, hop_length=CHROMA_HOP)[0]
    frame_rms = librosa.feature.rms(y=y, frame_length=4096, hop_length=CHROMA_HOP)[0]
    audible = frame_rms > 0.01 * frame_rms.max()  # ignore near-silent frames (no brightness to speak of)
    brightness = float(np.median(centroid[audible])) if audible.any() else 0.0
    return EnergyFeatures(rms_db(audio), len(onsets) / (len(y) / ANALYSIS_SR), brightness)


def energy_scores(features: list[EnergyFeatures | None]) -> list[float | None]:
    """Combine each song's features into one 0..1 energy score, relative to the set."""
    known = [f for f in features if f is not None]
    if not known:
        return [None] * len(features)
    columns = {
        "loudness": [f.loudness_db for f in known],
        "onset_rate": [f.onset_rate for f in known],
        "brightness": [math.log2(max(f.brightness_hz, 20.0)) for f in known],
    }
    scaled = {}
    for name, values in columns.items():
        lo, hi = min(values), max(values)
        spread = max(hi - lo, ENERGY_MIN_SPREAD[name])
        middle = (lo + hi) / 2
        scaled[name] = [float(np.clip(0.5 + (v - middle) / spread, 0.0, 1.0)) for v in values]
    combined = iter([sum(ENERGY_WEIGHTS[name] * scaled[name][i] for name in columns) for i in range(len(known))])
    return [next(combined) if f is not None else None for f in features]


# Everything about one song
# -------------------------


@dataclass(frozen=True)
class SongAnalysis(BeatAnalysis):
    """A song's beats, plus its key, energy, downbeats and structure."""

    key: str | None = None  # e.g. "A minor"
    key_confidence: float = 0.0
    key_method: str | None = None  # which detector found the key; None if not analyzed
    energy: EnergyFeatures | None = None
    downbeats: Downbeats | None = None
    structure: object = None  # structure.StructureFeatures (imported there, to avoid a circular import)
    beat_method: str = "ours"  # "ours" (detect_beats) or "beat_this"


# Caching
# -------
# Bump CACHE_VERSION whenever the analysis changes, so old results are redone.
# 5 added key and energy (3 and 4 were development versions of the same); 6 added
# downbeats and structure; 7 fixed tresillo songs read at 4/3 of their tempo.
CACHE_VERSION = 7
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "analysis_cache.json"


class AnalysisCache:
    """Song analyses saved in a JSON file, so each song is only analyzed once.

    Entries are keyed by each song's full path and remember the file's modification
    time. Edit or replace a song and its old entry stops matching, so the song gets
    analyzed again. Delete the JSON file to force everything to be re-analyzed.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else DEFAULT_CACHE_PATH
        self.entries = self._read()

    def get(self, song: str | Path) -> SongAnalysis | None:
        entry = self.entries.get(self._key(song))
        try:
            if entry is None or entry["mtime_ns"] != os.stat(song).st_mtime_ns:
                return None
            from structure import StructureFeatures

            energy, downbeats, structure = entry.get("energy"), entry.get("downbeats"), entry.get("structure")
            return SongAnalysis(
                entry["bpm"], np.array(entry["beats"], dtype=float),
                key=entry.get("key"), key_confidence=float(entry.get("key_confidence", 0.0)),
                key_method=entry.get("key_method"),
                energy=EnergyFeatures(**energy) if energy is not None else None,
                downbeats=None if downbeats is None else Downbeats(
                    np.array(downbeats["times"], dtype=float), int(downbeats["beats_per_bar"]), downbeats["method"]),
                structure=None if structure is None else StructureFeatures.from_dict(structure),
                beat_method=entry.get("beat_method", "ours"),
            )
        except (KeyError, TypeError, ValueError):
            return None  # a damaged entry: analyze the song again

    def put(self, song: str | Path, analysis: BeatAnalysis) -> None:
        energy = getattr(analysis, "energy", None)
        self.entries[self._key(song)] = {
            "mtime_ns": os.stat(song).st_mtime_ns,
            "bpm": analysis.bpm,
            "beats": [round(float(t), 5) for t in analysis.beats],
            "key": getattr(analysis, "key", None),
            "key_confidence": round(float(getattr(analysis, "key_confidence", 0.0)), 4),
            "key_method": getattr(analysis, "key_method", None),
            "energy": None if energy is None else {
                "loudness_db": round(energy.loudness_db, 3),
                "onset_rate": round(energy.onset_rate, 4),
                "brightness_hz": round(energy.brightness_hz, 1),
            },
        }
        downbeats, structure = getattr(analysis, "downbeats", None), getattr(analysis, "structure", None)
        if downbeats is not None:
            self.entries[self._key(song)]["downbeats"] = {
                "times": [round(float(t), 5) for t in downbeats.times],
                "beats_per_bar": downbeats.beats_per_bar, "method": downbeats.method,
            }
        if structure is not None:
            self.entries[self._key(song)]["structure"] = structure.to_dict()
        self.entries[self._key(song)]["beat_method"] = getattr(analysis, "beat_method", "ours")
        self._write()

    @staticmethod
    def _key(song: str | Path) -> str:
        return str(Path(song).resolve())

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}  # no cache yet, or an unreadable one: start fresh
        if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
            return {}  # made by a different version of the analysis
        songs = data.get("songs")
        return songs if isinstance(songs, dict) else {}

    def _write(self) -> None:
        data = {"version": CACHE_VERSION, "songs": self.entries}
        temp = self.path.with_name(self.path.name + ".tmp")
        try:
            temp.write_text(json.dumps(data))
            os.replace(temp, self.path)  # swap in the new file all at once
        except OSError as exc:
            print(f"  warning: couldn't save the analysis cache ({exc})", file=sys.stderr)
