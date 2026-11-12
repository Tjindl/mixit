"""Tempo matching: change a song's tempo without changing its pitch.

Two ways to use it (automix's --tempo-mode):

    global   pick one tempo for the whole mix and stretch every song to it, start
             to finish (Stage 2): time_stretch()
    ramp     keep each song at its own tempo and only change it around transitions,
             gliding smoothly to a shared tempo for the overlap and back again
             afterwards: transition_tempo(), TimeMap, apply_time_map()

The first half of this file explains constant stretches and the phase vocoder; the
second half ("Changing tempo gradually") explains time maps and ramps.

Why the stretch rate is target_bpm / song_bpm
---------------------------------------------
BPM counts beats per minute. Play a song r times as fast and everything in it,
beats included, happens r times as often:

    new BPM = r * song BPM

We want new BPM = target, so r = target / song. A 124 BPM song in a 128 BPM mix
needs r = 128 / 124 = 1.032: play it 3.2% faster. A beat at t seconds then arrives
at t / r (everything happens sooner), which is how we move the detected beats along
with the audio instead of detecting them all over again. "Stretch %" in the output
is (r - 1) * 100: +3.2% means sped up, a negative number means slowed down.

Changing speed without changing pitch: the phase vocoder
--------------------------------------------------------
Simply playing audio faster raises its pitch too (a record at the wrong speed),
because every cycle of every waveform gets shorter. A phase vocoder changes how long
things last while keeping their frequencies:

1. Analysis. Cut the audio into short overlapping windows and take the FFT of each.
   Every window becomes a list of frequency components, each with a magnitude (how
   loud it is) and a phase (where in its cycle the wave is).
2. Stretching. Build a new sequence of windows that walks through the original
   at a different speed: output window k is taken from position k * rate in the
   original, blending the two nearest windows' magnitudes. Which frequencies are
   playing, and how loudly, stays the same; only how long each moment lasts
   changes.
3. The phase fix. Each component is a spinning wave. Repeat or skip windows and the
   waves no longer join up from one window to the next, which you'd hear as
   warbling. So the vocoder measures how far each component's phase advances
   between neighbouring input windows (that advance IS its exact frequency) and
   adds the same advance at every output window. Every frequency keeps spinning at
   its original speed, and that is what keeps the pitch unchanged.
4. Synthesis. Inverse-FFT every new window and overlap-add them back into audio.

The weakness: step 3 treats every frequency separately, and loses the alignment
between their phases that makes a drum hit sharp. Hits get smeared across a window,
and noisy sounds (hi-hats, cymbals) lose some energy and sound "phasey". Rubber
Band detects drum hits and resets the phases there, and keeps neighbouring
frequencies' phases locked together, which is why it sounds noticeably better.
"""

import functools
import importlib.util
import math
import shutil
import statistics
import subprocess
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import librosa
import numpy as np

from audio_io import SAMPLE_RATE
from processing import peak

MAX_STRETCH = 0.08  # by default, songs needing more than an 8% tempo change aren't stretched

# Songs within 0.02% of the mix tempo aren't stretched at all. Over a 16-beat
# crossfade that's under 2 ms of drift, and skipping the stretch saves time and its
# side effects. It's common: many dance tracks share an exact tempo like 124 BPM.
NO_STRETCH_BELOW = 0.0002

# Rubber Band options. --centre-focus processes the two channels together, which
# keeps centre-panned sounds (kick, bass, vocals) focused. Add "--fine" to use the
# higher-quality R3 engine; it's about 6 times slower.
RUBBERBAND_FLAGS = ("--centre-focus",)

# Phase vocoder window: 1024 samples (23 ms) instead of librosa's default 2048. A
# shorter window smears drum hits less in time (on test click tracks, ~4 ms instead
# of ~10 ms), at the cost of resolving the lowest notes less precisely. For beat
# matching, sharp timing matters more.
VOCODER_WINDOW = 1024


def target_bpm(bpms: list[float | None]) -> float | None:
    """The shared tempo for the mix: the median of the songs' tempos.

    Why the median rather than the average? It's the middle song's tempo, so one
    oddball (a 90 BPM song in a set of 128s) can't drag the whole mix toward it.
    Songs with no tempo (None) are ignored; returns None if no song has one.
    """
    known = [bpm for bpm in bpms if bpm]
    return statistics.median(known) if known else None


def stretch_rate(song_bpm: float, target: float) -> float:
    """How many times faster (> 1) or slower (< 1) to play a song to reach `target` BPM."""
    return target / song_bpm


def stretch_beat_times(beats: np.ndarray, rate: float) -> np.ndarray:
    """Where beats land after stretching by `rate`: a beat at t seconds moves to t / rate."""
    return beats / rate


def rubberband_available() -> bool:
    """True if pyrubberband can work: the package is installed AND so is the program."""
    return (shutil.which("rubberband") is not None
            and importlib.util.find_spec("pyrubberband") is not None)


def stretch_method() -> str:
    """Which time-stretcher time_stretch() will use, for messages."""
    return "Rubber Band" if rubberband_available() else "librosa's phase vocoder"


def time_stretch(audio: np.ndarray, rate: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Play `audio` `rate` times as fast without changing its pitch.

    `audio` is float32 with shape (n, 2); the result has about n / rate samples.
    Uses Rubber Band through pyrubberband if it's installed, and librosa's phase
    vocoder otherwise.
    """
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    if abs(rate - 1.0) < 1e-6:
        return audio

    if rubberband_available():
        import pyrubberband

        # pyrubberband writes the audio to a temporary WAV file, runs the
        # `rubberband` program on it and reads the result back. It takes
        # samples-first arrays like ours, so (n, 2) goes in as it is.
        #
        # That temporary file is 16-bit, which can't hold values beyond +/-1.0, and
        # a loudness-normalized song can peak above 1.0. So scale the audio down to a
        # peak of 0.5 on the way in and back up on the way out; otherwise those peaks
        # would be clipped into distortion.
        scale = max(peak(audio), 1e-9) / 0.5
        # A fresh dict every call: pyrubberband writes the rate into the dict it's
        # given, so reusing one would silently reuse the first song's rate.
        options = {flag: "" for flag in RUBBERBAND_FLAGS}
        stretched = pyrubberband.time_stretch(audio / scale, sr, rate, rbargs=options) * scale
    else:
        # Each channel is cut into windows at the same moments and walked through at
        # the same speed, so left and right stay in sync. We stretch them one at a
        # time: the phase vocoder treats channels independently anyway, so the result
        # is the same, but it needs half the memory (its working arrays are large:
        # about 1.2 GB per channel of a 4-minute song).
        with warnings.catch_warnings():
            # librosa 1.0's time_stretch sets off two harmless warnings inside
            # librosa itself: a deprecated-parameter notice it passes to its own
            # phase_vocoder, and a numba cast notice. The output is unaffected.
            warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
            warnings.filterwarnings("ignore", message="invalid value encountered in cast")
            channels = audio if audio.ndim == 2 else audio[:, np.newaxis]
            stretched = np.column_stack([
                librosa.effects.time_stretch(
                    channels[:, c], rate=rate, n_fft=VOCODER_WINDOW, hop_length=VOCODER_WINDOW // 4
                )
                for c in range(channels.shape[1])
            ])
    return np.ascontiguousarray(stretched, dtype=np.float32)


# Changing tempo gradually (--tempo-mode ramp)
# ============================================
#
# Tempo is the slope of beat position over time
# ---------------------------------------------
# Count beats as the song plays: B(t) is how many beats have gone by at time t. For a
# song with a steady tempo, B(t) is a straight line, and its slope is the tempo:
#
#     tempo (beats per second) = dB/dt          BPM = 60 * dB/dt
#
# That's the definition of a derivative: how fast one thing (beats) changes as
# another (time) moves on. A steeper line means more beats per second, a faster song.
#
# What a time map is
# ------------------
# A time-stretch decides, for every moment of the OUTPUT, which moment of the ORIGINAL
# song is playing. Call that input position t(tau), where tau is output time. A
# constant stretch at rate r is the straight line t = r * tau. A time map is any
# increasing curve, given as a list of anchor points (input sample, output sample)
# joined by straight lines: "input sample 441000 must come out at output sample
# 430000". Rubber Band's --timemap option takes exactly such a list.
#
# The chain rule tells us what a time map does to the tempo. The beats heard by
# output time tau are B(t(tau)), so the tempo we hear is
#
#     d/dtau B(t(tau)) = B'(t) * t'(tau) = original tempo * rate
#
# where the rate t'(tau) is the time map's slope: seconds of the original played per
# second of output. Slope 1 means the original tempo; slope 1.03 means 3% faster.
# So to shape the tempo, we shape the slope of the time map.
#
# Why ramp instead of jump
# ------------------------
# We could switch from 120 to 123 BPM in an instant, at a beat. But a listener locks
# on to the gap between beats (500 ms at 120 BPM) and predicts the next one; after a
# sudden switch the next beat arrives 12 ms "early", and a steady pulse that slips
# like that is noticed as a stumble. Spread the same change over 16 beats and each
# gap is at most ~1 ms shorter than the one before it: far too little to notice from
# one beat to the next, so the tempo just seems to drift, the way a DJ nudges a
# pitch fader.
#
# The shape of the ramp matters at its ends too. A straight-line ramp starts and stops
# changing abruptly (a "corner" in the tempo). We use smoothstep instead,
#
#     s(u) = 3u^2 - 2u^3        u = 0 -> 1 over the ramp
#
#     rate(tau) = r0 + (r1 - r0) * s(tau / T)
#
# which starts and ends with zero slope, so the tempo eases out of one value and into
# the next. Integrating the rate gives the input position the time map needs:
#
#     t(tau) = r0 * tau + (r1 - r0) * T * (u^3 - u^4 / 2)
#
# At the end of the ramp (u = 1) that's T * (r0 + r1) / 2, so a ramp that covers L
# samples of the original lasts T = 2L / (r0 + r1) samples in the output: the
# original length divided by the average rate, as you'd expect.

MEET_MODES = ("middle", "incoming")
RAMP_ANCHORS_PER_BEAT = 4  # time-map anchors per beat of a ramp: the tempo moves in tiny steps


@dataclass(frozen=True)
class TransitionTempo:
    """The shared tempo two neighbouring songs meet at for a ramp-mode transition."""

    bpm: float  # tempo of the overlap, counted in the outgoing song's beats
    rate_out: float  # how much faster (> 1) the outgoing song plays during the overlap
    rate_in: float  # the same for the incoming song
    factor: float = 1.0  # incoming BPM x factor (0.5, 1 or 2) = the tempo it's matched as

    @property
    def stretch(self) -> float:
        """The bigger of the two songs' tempo changes, as a fraction (0.03 = 3%)."""
        return max(abs(self.rate_out - 1), abs(self.rate_in - 1))


def octave_match(bpm_out: float, bpm_in: float) -> float:
    """The factor (0.5, 1 or 2) that brings `bpm_in` nearest to `bpm_out`.

    An 85 BPM song mixes into a 170 BPM one by matching two of its beats to every
    beat of the other; then the factor is 2 (85 is matched as 170). Usually it's 1,
    because analysis.octave_factors has already fixed half/double readings.
    """
    return min((0.5, 1.0, 2.0), key=lambda f: abs(math.log(bpm_in * f / bpm_out)))


def transition_tempo(bpm_out: float, bpm_in: float, meet: str = "middle") -> TransitionTempo:
    """Where an outgoing and an incoming song meet, and how much each must change.

    meet="incoming": the outgoing song speeds up or slows down to the incoming
        song's tempo; the incoming song starts untouched.
    meet="middle": both move halfway. "Halfway" is the geometric mean
        sqrt(a * b), which makes both songs change by the same RATIO: 120 and 126
        meet at 122.96, with 120 x 1.0247 = 122.96 = 126 / 1.0247. Each song then
        stretches by about half as much as with "incoming".
    """
    if meet not in MEET_MODES:
        raise ValueError(f"meet must be one of {MEET_MODES}, got {meet!r}")
    factor = octave_match(bpm_out, bpm_in)
    matched_in = bpm_in * factor
    bpm = matched_in if meet == "incoming" else math.sqrt(bpm_out * matched_in)

    def snap(rate: float) -> float:  # a change too small to matter isn't worth a stretch
        return 1.0 if abs(rate - 1) < NO_STRETCH_BELOW else rate

    return TransitionTempo(bpm, snap(bpm / bpm_out), snap(bpm / matched_in), factor)


@dataclass(frozen=True)
class TimeMap:
    """Where each sample of a song ends up after a variable time-stretch.

    Anchor k says input sample src[k] comes out at output sample dst[k]; in between,
    positions are joined by straight lines (constant rate), just as Rubber Band
    reads its --timemap file. Both arrays are integers and strictly increasing,
    starting at (0, 0) and ending at (length of the song, length of the output).
    """

    src: np.ndarray
    dst: np.ndarray

    def __call__(self, positions):
        """Input sample positions -> output positions (floats). Beat times go through here."""
        return np.interp(positions, self.src, self.dst)

    def inverse(self, positions):
        """Output positions -> the input positions that play there."""
        return np.interp(positions, self.dst, self.src)

    def rate_at(self, position: float) -> float:
        """The rate (slope of input against output) at input `position`."""
        k = int(np.clip(np.searchsorted(self.src, position, side="right") - 1, 0, len(self.src) - 2))
        return float((self.src[k + 1] - self.src[k]) / (self.dst[k + 1] - self.dst[k]))

    def stretched_regions(self, start: int, end: int, min_gap: int = 0) -> list[tuple[int, int]]:
        """Input ranges within [start, end) where the rate isn't exactly 1.

        Regions closer together than `min_gap` are merged into one, because the
        untouched audio between them is too short to be worth stitching back in.
        """
        changed = np.diff(self.src) != np.diff(self.dst)
        regions: list[list[int]] = []
        for k in np.flatnonzero(changed):
            a, b = max(int(self.src[k]), start), min(int(self.src[k + 1]), end)
            if a >= b:
                continue
            if regions and a - regions[-1][1] < max(min_gap, 1):
                regions[-1][1] = b
            else:
                regions.append([a, b])
        return [(a, b) for a, b in regions]


class TimeMapBuilder:
    """Builds a TimeMap one stretch of the song at a time, from the start onwards."""

    def __init__(self) -> None:
        self.src, self.dst = [0], [0]
        self._dst_exact = 0.0  # unrounded output position, so rounding never accumulates

    def hold(self, until: float, rate: float) -> None:
        """Play at a constant `rate` up to input sample `until`."""
        length = round(until) - self.src[-1]
        if length > 0:
            self._add(self.src[-1] + length, self._dst_exact + length / rate)

    def ramp(self, until: float, rate_from: float, rate_to: float, n_anchors: int) -> None:
        """Glide from `rate_from` to `rate_to` (smoothstep) by input sample `until`."""
        start_src, start_dst = self.src[-1], self._dst_exact
        length = round(until) - start_src
        if length <= 0:
            return
        if rate_from == rate_to:
            self.hold(until, rate_from)
            return
        duration = 2 * length / (rate_from + rate_to)  # output length of the ramp (see above)
        for u in np.linspace(0.0, 1.0, max(2, n_anchors) + 1)[1:]:
            src = start_src + duration * (rate_from * u + (rate_to - rate_from) * (u**3 - u**4 / 2))
            self._add(src, start_dst + duration * u)

    def _add(self, src: float, dst: float) -> None:
        s, d = round(src), round(dst)
        if s > self.src[-1] and d > self.dst[-1]:  # anchors must keep moving forwards
            self.src.append(s)
            self.dst.append(d)
            self._dst_exact = dst

    def build(self) -> TimeMap:
        return TimeMap(np.array(self.src, dtype=np.int64), np.array(self.dst, dtype=np.int64))


# Stretching only what needs it
# -----------------------------
# A ramp-mode song is mostly untouched: only its intro (the overlap with the song
# before, then a ramp back to its own tempo) and its outro (a ramp to the next
# song's tempo, then the overlap) are stretched. apply_time_map() runs Rubber Band
# on just those regions and copies everything else across sample for sample.
#
# The joins need care. Each stretched chunk runs on into the untouched audio on
# both sides, at rate 1. There, Rubber Band's R3 engine reproduces the original
# almost perfectly (measured on real songs: -70 to -80 dB of difference), except
# within ~50 ms of where the rate starts or stops changing, because its analysis
# window reaches across the change. So each join sits SETTLE (100 ms) out from the
# stretched region, where the two agree.
#
# The stretched audio can also land a few samples (~0.3 ms) away from where the
# time map says. We find that offset by cross-correlation (slide one signal past
# the other, multiplying and adding at each offset, and keep the offset where they
# match best) and join there, with a short crossfade.
#
# That crossfade is LINEAR, not equal-power: the two signals being joined are the
# same audio, so their amplitudes add rather than their powers (see transitions.py).
# Linear gains that sum to 1 keep the level steady for identical signals.
#
# So everything outside the ramps and overlaps is copied sample for sample, except
# SETTLE + STITCH (~0.1 s) beside each stretched region.

TIMEMAP_RUBBERBAND_FLAGS = ("--centre-focus", "--fine")  # R3 engine: see ramp_stretch_available
SETTLE = 4410  # samples (100 ms) between a stretched region and the join back to the original
STITCH = 256  # samples of crossfade (~6 ms) at that join
LAG_WINDOW = 4096  # samples compared when measuring how far off the stretched audio landed
MAX_LAG = 512  # the furthest (in samples) we search for that offset
CONTEXT = 4410  # extra samples beyond all that, for Rubber Band's start-up and wind-down
MARGIN = SETTLE + LAG_WINDOW + MAX_LAG + CONTEXT  # how far a chunk reaches past its region
MIN_UNTOUCHED = 2 * MARGIN  # shorter untouched stretches are stretched along with their neighbours

Stretcher = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]


def ramp_stretch_available() -> bool:
    """True if Rubber Band is installed and new enough (3.0+) for ramp mode.

    Ramp mode needs Rubber Band's R3 engine (--fine). Measured on real songs, the
    older R2 engine drifts ~5 ms from the time map over a stretch, and even at rate
    1 its output differs from the original (it re-synthesizes every phase), so
    stretched chunks couldn't be joined seamlessly to untouched audio. R3 stays
    within ~1 ms of the map, and at rate 1 reproduces the original almost exactly.
    """
    path = shutil.which("rubberband")
    return rubberband_available() and path is not None and _has_r3_engine(path)


@functools.lru_cache(maxsize=4)
def _has_r3_engine(path: str) -> bool:
    try:
        result = subprocess.run([path, "--help"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    return "--fine" in result.stdout + result.stderr


def stretch_time_map(audio: np.ndarray, src: np.ndarray, dst: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Time-stretch `audio` along a time map with Rubber Band (R3 engine).

    `src` and `dst` are the anchors, starting at (0, 0) and ending at
    (len(audio), output length). Returns float32 audio of exactly dst[-1] samples.
    """
    import pyrubberband

    # As in time_stretch(): scale to a 0.5 peak for Rubber Band's 16-bit temporary
    # files, so peaks above 1.0 aren't clipped, and back afterwards.
    scale = max(peak(audio), 1e-9) / 0.5
    options = {flag: "" for flag in TIMEMAP_RUBBERBAND_FLAGS}
    # Leave out the (0, 0) anchor: the start is implied, and in testing an explicit
    # (0, 0) made R3 ignore the map for the first ~2 seconds when the rate there
    # wasn't 1 (beats landed up to 40 ms early). Without it, they're within 1 ms.
    pairs = [(s, d) for s, d in zip(src.tolist(), dst.tolist()) if s > 0]
    stretched = pyrubberband.timemap_stretch(audio / scale, sr, pairs, rbargs=options) * scale
    return _fit_length(np.asarray(stretched, dtype=np.float32), int(dst[-1]))


@dataclass
class TimeMappedTrack:
    """The played part of a song after apply_time_map(), and where things landed in it."""

    audio: np.ndarray
    time_map: TimeMap
    out_start: int  # the output position (in time-map terms) where `audio` begins
    untouched: list[tuple[int, int, int]] = field(default_factory=list)  # (input start, end, track position)
    shifts: list[tuple[int, int]] = field(default_factory=list)  # (input position, samples late from there on)

    def positions(self, input_positions) -> np.ndarray:
        """Input sample positions (beats, say) -> sample positions in `audio`.

        This is how beat times follow the stretch: through the same time map the
        audio went through, instead of detecting the beats all over again.
        """
        input_positions = np.asarray(input_positions, dtype=float)
        result = self.time_map(input_positions) - self.out_start
        for start, shift in self.shifts:  # each shift holds from its position onwards
            result = np.where(input_positions >= start, self.time_map(input_positions) - self.out_start + shift,
                              result)
        return result


def apply_time_map(
    audio: np.ndarray,
    time_map: TimeMap,
    out_start: int,
    out_end: int,
    stretcher: Stretcher | None = None,
    sr: int = SAMPLE_RATE,
) -> TimeMappedTrack:
    """Stretch `audio` along `time_map`, keeping only output samples out_start..out_end.

    Regions where the map's rate is exactly 1 are copied straight from `audio`;
    only the rest goes through `stretcher` (Rubber Band, unless the tests pass a
    different one). See "Stretching only what needs it" above.
    """
    stretcher = stretcher or stretch_time_map
    n = len(audio)
    in_start = int(np.clip(np.floor(time_map.inverse(out_start) + 1e-6), 0, n))
    in_end = int(np.clip(np.ceil(time_map.inverse(out_end) - 1e-6), 0, n))
    track = TimeMappedTrack(np.zeros((0, *audio.shape[1:]), np.float32), time_map, out_start)
    pieces: list[np.ndarray] = []
    emitted = 0  # samples of the track so far

    def emit(piece: np.ndarray) -> None:
        nonlocal emitted
        pieces.append(piece)
        emitted += len(piece)

    shift = 0  # how many samples later than the map says the untouched audio now sits
    cursor: int | None = in_start  # next input sample to copy, or None once a chunk runs to the end
    for start, end in _regions_to_stretch(time_map, in_start, in_end):
        lo, hi = max(0, start - MARGIN), min(n, end + MARGIN)
        chunk, d0 = _stretch_chunk(audio, time_map, lo, hi, stretcher, sr)

        if start > in_start:  # untouched audio before this chunk: copy it, then join
            assert cursor is not None
            join = start - SETTLE  # the chunk takes over from here
            if join - STITCH > cursor:
                track.untouched.append((cursor, join - STITCH, emitted))
                emit(audio[cursor : join - STITCH])
            lag_in = _find_lag(chunk, round(time_map(join - LAG_WINDOW)) - d0, audio, join - LAG_WINDOW)
            i = round(time_map(join - STITCH)) - d0 + lag_in  # chunk index holding input sample join - STITCH
            emit(_linear_crossfade(audio[join - STITCH : join], _take(chunk, i, STITCH)))
            i += STITCH
        else:  # the chunk is where the track begins
            lag_in = 0
            i = out_start - d0

        if end < in_end:  # untouched audio after this chunk: join back to it
            join = end + SETTLE
            lag_out = _find_lag(chunk, round(time_map(join)) - d0, audio, join)
            j = round(time_map(join)) - d0 + lag_out  # chunk index holding input sample `join`
            emit(_take(chunk, i, j - i))
            emit(_linear_crossfade(_take(chunk, j, STITCH), audio[join : join + STITCH]))
            shift += lag_out - lag_in
            track.shifts.append((join, shift))
            cursor = join + STITCH
        else:  # the chunk runs to the end of the track
            emit(_take(chunk, i, out_end - d0 + lag_in - i))
            cursor = None

    if cursor is not None and in_end > cursor:
        track.untouched.append((cursor, in_end, emitted))
        emit(audio[cursor:in_end])
    planned = out_end - out_start + shift
    if emitted < planned:  # asked for more than the song has: the rest is silence
        emit(np.zeros((planned - emitted, *audio.shape[1:]), np.float32))
    track.audio = np.ascontiguousarray(np.concatenate(pieces) if pieces else track.audio, dtype=np.float32)
    return track


def _regions_to_stretch(time_map: TimeMap, in_start: int, in_end: int) -> list[tuple[int, int]]:
    """The stretched regions of the played part, with any too close to an end run out to it.

    Joining back to the original needs SETTLE + LAG_WINDOW samples of untouched audio
    beside the region; when the track's start or end is closer than that, the chunk
    simply starts or ends the track instead.
    """
    near = SETTLE + LAG_WINDOW + STITCH
    return [(in_start if start - in_start < near else start, in_end if in_end - end < near else end)
            for start, end in time_map.stretched_regions(in_start, in_end, MIN_UNTOUCHED)]


def _stretch_chunk(audio, time_map: TimeMap, lo: int, hi: int, stretcher: Stretcher, sr: int):
    """Stretch input samples lo..hi. Returns (output, d0): output[0] is map position d0."""
    d0, d1 = round(float(time_map(lo))), round(float(time_map(hi)))
    # The map's own anchors between the chunk's ends, minus any that rounding the ends
    # to whole samples has tied with them.
    inside = (time_map.src > lo) & (time_map.src < hi) & (time_map.dst > d0) & (time_map.dst < d1)
    src = np.concatenate([[lo], time_map.src[inside], [hi]]) - lo
    dst = np.concatenate([[d0], time_map.dst[inside], [max(d1, d0 + 1)]]) - d0
    return _fit_length(stretcher(audio[lo:hi], src, dst, sr), int(dst[-1])), d0


def _find_lag(chunk: np.ndarray, index: int, audio: np.ndarray, t: int, length: int = LAG_WINDOW) -> int:
    """How many samples later than `index` the input samples t..t+length sit in `chunk`.

    Cross-correlation: slide the original along the stretched audio, multiply and add
    at each offset, and keep the offset where they agree best. Returns 0 when there's
    nothing to go on (silence, or no offset gives a convincing match).
    """
    lo, hi = index - MAX_LAG, index + length + MAX_LAG
    if lo < 0 or hi > len(chunk) or t < 0 or t + length > len(audio):
        return 0
    ref = audio[t : t + length].mean(axis=1) if audio.ndim == 2 else audio[t : t + length]
    seg = chunk[lo:hi].mean(axis=1) if chunk.ndim == 2 else chunk[lo:hi]
    corr = np.correlate(seg.astype(np.float64), ref.astype(np.float64), mode="valid")
    k = int(np.argmax(corr))
    match = seg[k : k + length]
    norm = math.sqrt(float(np.dot(ref, ref)) * float(np.dot(match, match)))
    return k - MAX_LAG if norm > 0 and corr[k] / norm > 0.5 else 0


def _linear_crossfade(outgoing: np.ndarray, incoming: np.ndarray) -> np.ndarray:
    """Join two copies of the same audio: gains 1 - t and t, which add up to 1."""
    t = np.linspace(0.0, 1.0, len(outgoing), dtype=np.float32)
    if outgoing.ndim == 2:
        t = t[:, np.newaxis]
    return (outgoing * (1 - t) + incoming * t).astype(np.float32)


def _take(audio: np.ndarray, start: int, length: int) -> np.ndarray:
    """audio[start : start + length], zero-padded if that runs off either end."""
    length = max(0, length)
    out = np.zeros((length, *audio.shape[1:]), dtype=np.float32)
    a, b = max(0, start), min(len(audio), start + length)
    if b > a:
        out[a - start : b - start] = audio[a:b]
    return out


def _fit_length(audio: np.ndarray, length: int) -> np.ndarray:
    """Trim or zero-pad `audio` to exactly `length` samples."""
    if len(audio) >= length:
        return audio[:length]
    return np.concatenate([audio, np.zeros((length - len(audio), *audio.shape[1:]), dtype=audio.dtype)])
