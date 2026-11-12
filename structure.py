"""Song structure: bars, phrases and sections (Stage 4).

To mix mid-song like a DJ, automix has to know where a song's parts are: the intro,
the builds, the drops, the breakdowns, the outro. This module finds them in three
steps:

    1. features    for every beat: what notes are playing (chroma), what it sounds
                   like (MFCC timbre), how loud it is, how much bass it has
    2. boundaries  where the song changes character: a self-similarity matrix and
                   Foote's checkerboard novelty (explained below), snapped to phrases
    3. labels      which section is which, from its energy, its bass and where it is

analyze_structure() does the expensive part once per song (the result is cached),
and sections() turns it into labelled sections for a given phrase length.

Phrases: why dance music comes in 8-bar blocks
----------------------------------------------
Dance music is built for DJs and dancers, so it's extremely regular. Bars group into
phrases of 8 bars (32 beats), and phrases into sections of 8, 16 or 32 bars: a new
element (hi-hats, a bass line, a vocal) comes in or drops out at the start of a
phrase, and builds and drops land on phrase boundaries. Listeners feel that grid
without counting it; a change that comes 2 bars "early" sounds wrong. So DJs start
and end transitions on phrase boundaries, and so do we: every section boundary is
moved to the nearest phrase boundary (--phrase-bars sets the phrase length).
"""

import hashlib
import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

import analysis
from analysis import ANALYSIS_SR, CHROMA_LOWEST, CHROMA_OCTAVES, Downbeats, key_from_chroma
from audio_io import SAMPLE_RATE

FEATURE_HOP = 512  # feature frames every 23 ms at 22.05 kHz, averaged per beat
BASS_CUTOFF = 150.0  # Hz: "bass energy" is everything below this
NOVELTY_HALF_WIDTH = 16  # beats on each side of the checkerboard kernel (4 bars)
PEAK_WINDOW = 8  # a novelty peak must be the highest within this many beats either side
PEAK_THRESHOLD = 0.3  # ...and at least this fraction of the song's biggest change

# Section labels, from their intensity (loudness and bass, 0 = the song's quiet parts, 1 = its loud ones):
DROP_ENERGY = 0.65  # at least this intense: a drop (or chorus)
BREAKDOWN_ENERGY = 0.35  # at most this intense, in the middle of the song: a breakdown
BUILD_RISE_DB = 2.0  # rising by at least this much across the section (dB), into a drop: a build
FLAT_SONG_DB = 1.5  # if the song's loud and quiet beats are this close, energy can't tell sections apart

LABELS = ("intro", "build", "drop", "breakdown", "groove", "outro")


@dataclass(frozen=True)
class StructureFeatures:
    """The per-beat measurements a song's structure is worked out from (cached)."""

    beat_times: np.ndarray  # the beats everything is measured at (seconds, file time)
    downbeats: np.ndarray  # bar starts (seconds)
    beats_per_bar: int
    energy_db: np.ndarray  # loudness of each beat (RMS, dB)
    bass_db: np.ndarray  # loudness below BASS_CUTOFF (dB)
    novelty: np.ndarray  # how much the music changes at each beat (0..1)
    chroma: np.ndarray  # (beats, 12): the notes playing at each beat
    vocal_db: np.ndarray | None = None  # loudness of the vocals at each beat (demucs), if known
    segments: tuple | None = None  # ((start, end, label), ...) from allin1, if it was used
    downbeat_method: str = "bass"
    segment_method: str = "foote"

    def to_dict(self) -> dict:
        """Plain lists and numbers, for the JSON analysis cache."""
        def rounded(array, digits):
            return None if array is None else np.round(np.asarray(array, dtype=float), digits).tolist()

        return {
            "beat_times": rounded(self.beat_times, 5), "downbeats": rounded(self.downbeats, 5),
            "beats_per_bar": self.beats_per_bar, "energy_db": rounded(self.energy_db, 2),
            "bass_db": rounded(self.bass_db, 2), "novelty": rounded(self.novelty, 3),
            "chroma": rounded(self.chroma, 3), "vocal_db": rounded(self.vocal_db, 2),
            "segments": [list(s) for s in self.segments] if self.segments else None,
            "downbeat_method": self.downbeat_method, "segment_method": self.segment_method,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StructureFeatures":
        def array(key, shape=None):
            value = data.get(key)
            return None if value is None else np.array(value, dtype=float).reshape(shape or -1)

        return cls(
            beat_times=array("beat_times"), downbeats=array("downbeats"), beats_per_bar=int(data["beats_per_bar"]),
            energy_db=array("energy_db"), bass_db=array("bass_db"), novelty=array("novelty"),
            chroma=array("chroma", (-1, 12)), vocal_db=array("vocal_db"),
            segments=tuple(tuple(s) for s in data["segments"]) if data.get("segments") else None,
            downbeat_method=data.get("downbeat_method", "bass"), segment_method=data.get("segment_method", "foote"),
        )


@dataclass(frozen=True)
class Section:
    """One part of a song: [start, end) in seconds (file time), starting on a phrase boundary."""

    start: float
    end: float
    label: str  # one of LABELS ("drop" also covers a pop song's chorus)
    number: int  # 1 for the first drop, 2 for the second...
    energy: float  # intensity: 0 (the song's quiet parts) to 1 (its loudest), from loudness and bass
    bass: float  # bass energy alone, on the same scale
    rise_db: float  # how much louder its last bars are than its first
    vocals: float | None  # 0 (instrumental) to 1 (vocals up front); None if unknown
    chroma: np.ndarray  # the section's average chroma vector (12 numbers)

    @property
    def name(self) -> str:
        """"drop 2", "build 1", "intro": numbered when a song has more than one of a kind."""
        return f"{self.label} {self.number}" if self.number else self.label

    @property
    def key(self):
        """The section's local key (analysis.KeyEstimate), from its chroma."""
        return key_from_chroma(self.chroma)


@dataclass(frozen=True)
class Structure:
    """A song's bars, phrases and sections, for one --phrase-bars setting."""

    downbeats: np.ndarray  # seconds
    phrases: np.ndarray  # phrase start times (seconds), each a downbeat
    beats_per_bar: int
    sections: list[Section]

    def section_at(self, time: float) -> Section | None:
        """The section playing at `time`."""
        for section in self.sections:
            if section.start <= time < section.end:
                return section
        return self.sections[-1] if self.sections and time >= self.sections[-1].end else None


# 1. Beat-synchronous features
# ============================
# A song has thousands of feature frames, but music changes in beats, so we average
# the frames between each beat and the next: one number (or vector) per beat. That
# makes the next step's matrix small (a 4-minute song at 128 BPM has ~500 beats), and
# lines it up with the musical grid.
#
#   chroma  the 12 pitch classes (see analysis.py): which chords are playing
#   MFCC    "mel-frequency cepstral coefficients": a compact description of the shape
#           of the spectrum, i.e. the timbre. Two passages with the same instruments
#           have similar MFCCs whatever notes they play.


def beat_features(audio: np.ndarray, beats: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, np.ndarray]:
    """Chroma, MFCC, loudness and bass loudness for each beat [beats[i], beats[i+1])."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    y = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=ANALYSIS_SR) if sr != ANALYSIS_SR else mono
    frames = librosa.time_to_frames(beats, sr=ANALYSIS_SR, hop_length=FEATURE_HOP)
    chroma = librosa.feature.chroma_cqt(y=y, sr=ANALYSIS_SR, hop_length=FEATURE_HOP,
                                        fmin=librosa.note_to_hz(CHROMA_LOWEST), n_octaves=CHROMA_OCTAVES)
    mfcc = librosa.feature.mfcc(y=y, sr=ANALYSIS_SR, hop_length=FEATURE_HOP, n_mfcc=14)[1:]  # 0th = loudness
    rms = librosa.feature.rms(y=y, hop_length=FEATURE_HOP)[0]
    bass = sosfiltfilt(butter(4, BASS_CUTOFF, "low", fs=ANALYSIS_SR, output="sos"), y)
    bass_rms = librosa.feature.rms(y=bass, hop_length=FEATURE_HOP)[0]

    def per_beat(feature):
        return beat_means(feature, frames)

    def db(x):
        return 20 * np.log10(np.maximum(x, 1e-6))

    return {
        "chroma": per_beat(chroma).T,
        "mfcc": per_beat(mfcc).T,
        "energy_db": db(per_beat(rms)[0]),
        "bass_db": db(per_beat(bass_rms)[0]),
    }


def beat_means(feature: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Average `feature` (dims x frames) over each beat: frames[i] up to frames[i+1] (the last to the end).

    Always one column per beat. (librosa.util.sync does the same job, but silently
    drops beats that fall past the last frame, which put every later beat off by one.)
    """
    feature = np.atleast_2d(feature)
    n = feature.shape[1]
    starts = np.clip(np.asarray(frames, dtype=int), 0, n - 1)
    ends = np.minimum(np.maximum(np.append(starts[1:], n), starts + 1), n)
    totals = np.concatenate([np.zeros((feature.shape[0], 1)), np.cumsum(feature, axis=1)], axis=1)
    return (totals[:, ends] - totals[:, starts]) / (ends - starts)


# 2. Where the music changes: self-similarity and novelty
# =======================================================
# The self-similarity matrix
#   Compare every beat with every other beat: S[i, j] = how similar beat i sounds to
#   beat j (here the cosine similarity of their feature vectors: 1 = identical).
#   Drawn as a picture, with time running along both axes, the diagonal is bright
#   (every beat is like itself), and each section shows up as a bright SQUARE on the
#   diagonal: within a drop, every beat is like every other. A repeated section (the
#   second drop) shows up as a bright square OFF the diagonal, next to the first.
#
#       time ->   intro   drop   break  drop
#        intro  [ ####                      ]
#        drop   [        ####         ####  ]
#        break  [               ####        ]
#        drop   [        ####         ####  ]
#
# The checkerboard kernel (Foote, 2000)
#   A section boundary is where one bright square ends and the next begins: just
#   before it, beats are like each other; just after it, beats are like each other;
#   but before and after aren't alike. Foote's kernel looks for exactly that. It's a
#   small 2x2 checkerboard,
#
#       +1  -1        "past like past" and "future like future" count for it,
#       -1  +1        "past like future" counts against it
#
#   (smoothed with a Gaussian so beats near the centre count most). Slide it along
#   the diagonal, multiply it with the patch of S under it and add everything up:
#   that's the novelty at that beat. It's high exactly at the corner between two
#   squares, i.e. where the music changes character. Its peaks are the boundaries.


def self_similarity(features: np.ndarray) -> np.ndarray:
    """Cosine similarity between every pair of rows (beats) of `features`."""
    unit = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-9)
    return unit @ unit.T


def checkerboard_kernel(half: int) -> np.ndarray:
    """Foote's kernel, 2*half square: +1 on the two diagonal quadrants, -1 off them, Gaussian-tapered."""
    t = np.arange(-half, half) + 0.5
    taper = np.exp(-0.5 * (t / (half / 2)) ** 2)
    signs = np.sign(t)
    return np.outer(signs * taper, signs * taper)


def novelty_curve(similarity: np.ndarray, half: int = NOVELTY_HALF_WIDTH) -> np.ndarray:
    """Slide the checkerboard along the diagonal: novelty[i] = change between beat i-1 and beat i (0..1)."""
    n = len(similarity)
    kernel = checkerboard_kernel(half)
    padded = np.pad(similarity, half, mode="edge")  # repeat the edges, so the ends don't look like changes
    novelty = np.array([np.sum(kernel * padded[i : i + 2 * half, i : i + 2 * half]) for i in range(n)])
    novelty = np.maximum(novelty, 0.0)
    return novelty / novelty.max() if novelty.max() > 0 else novelty


def analyze_structure(audio: np.ndarray, beats: np.ndarray, downbeats: Downbeats, sr: int = SAMPLE_RATE,
                      path: str | Path | None = None) -> StructureFeatures | None:
    """Measure everything structure needs, once per song. None if the song has too few beats."""
    if len(beats) < 4 * NOVELTY_HALF_WIDTH or len(downbeats.times) < 4:
        return None
    feats = beat_features(audio, beats, sr)
    # Standardize each feature dimension, so chroma and timbre count equally.
    combined = np.hstack([feats["chroma"], feats["mfcc"]])
    combined = (combined - combined.mean(axis=0)) / np.maximum(combined.std(axis=0), 1e-9)
    novelty = novelty_curve(self_similarity(combined))
    segments = _allin1_segments(path) if path is not None else None
    return StructureFeatures(
        beat_times=np.asarray(beats, dtype=float), downbeats=np.asarray(downbeats.times, dtype=float),
        beats_per_bar=downbeats.beats_per_bar, energy_db=feats["energy_db"], bass_db=feats["bass_db"],
        novelty=novelty, chroma=feats["chroma"],
        vocal_db=vocal_levels(path, beats) if path is not None else None,
        segments=segments, downbeat_method=downbeats.method,
        segment_method="allin1" if segments else "foote",
    )


# 3. Sections
# ===========


def phrase_starts(features: StructureFeatures, phrase_bars: int) -> np.ndarray:
    """Which downbeats start phrases: every `phrase_bars`-th, at the offset where the music changes most.

    The song's first downbeat isn't necessarily the start of a phrase (intros can
    have an odd number of bars), so we try each of the `phrase_bars` possible
    offsets and keep the one whose phrase starts have the most novelty in total.
    """
    beat_index = _nearest_beats(features.beat_times, features.downbeats)
    scores = [features.novelty[beat_index[offset::phrase_bars]].sum() for offset in range(min(phrase_bars, len(beat_index)))]
    offset = int(np.argmax(scores)) if scores else 0
    return features.downbeats[offset::phrase_bars]


def sections(features: StructureFeatures, phrase_bars: int = 8, duration: float | None = None) -> Structure:
    """Split the song into labelled sections whose boundaries are phrase boundaries."""
    phrases = phrase_starts(features, phrase_bars)
    start = float(features.beat_times[0])
    end = float(duration if duration is not None else features.beat_times[-1] + np.median(np.diff(features.beat_times)))
    if features.segments:  # allin1's boundaries, moved onto our phrase grid
        candidates = [s for s, _, _ in features.segments[1:]]
    else:  # novelty peaks
        n = features.novelty
        candidates = [features.beat_times[i] for i in range(1, len(n) - 1)
                      if n[i] >= PEAK_THRESHOLD and n[i] == n[max(0, i - PEAK_WINDOW) : i + PEAK_WINDOW + 1].max()]
    boundaries = sorted({_snap(t, phrases) for t in candidates})
    min_length = phrase_bars * features.beats_per_bar * np.median(np.diff(features.beat_times)) / 2
    boundaries = [b for b in boundaries if b - start >= min_length and end - b >= min_length]
    edges = [start, *boundaries, end]
    allin1_labels = _allin1_labels(features, edges) if features.segments else None
    return Structure(features.downbeats, phrases, features.beats_per_bar,
                     _label(features, edges, allin1_labels))


def _label(features: StructureFeatures, edges: list[float], given: list[str] | None) -> list[Section]:
    """Label each section from its energy, bass and position (see the rules below)."""
    beat_times = features.beat_times
    spans = [(beat_times >= a) & (beat_times < b) for a, b in zip(edges, edges[1:])]
    spans = [s if s.any() else (beat_times == beat_times[min(np.searchsorted(beat_times, a), len(beat_times) - 1)])
             for s, a in zip(spans, edges)]

    # Each section's loudness and bass, on a 0..1 scale running from the song's quiet
    # beats (10th percentile) to its loud ones (95th). Percentiles rather than the very
    # quietest and loudest moments, so a fade-out at the end doesn't squash everything
    # else together at the top.
    def scaled(per_beat):
        low, high = np.percentile(per_beat, 10), np.percentile(per_beat, 95)
        values = np.array([per_beat[s].mean() for s in spans])
        if high - low < FLAT_SONG_DB:
            return np.full(len(spans), 0.5), True
        return np.clip((values - low) / (high - low), 0.0, 1.0), False

    e, flat_energy = scaled(features.energy_db)
    b, flat_bass = scaled(features.bass_db)
    flat = flat_energy and flat_bass
    # Intensity: loudness and bass together. Bass tells sections apart better than
    # overall loudness in heavily compressed masters, where a verse can be within
    # 2 dB of the chorus but its bass line 10 dB quieter.
    intensity = (e + b) / 2

    def rise(s):
        values = features.energy_db[s]
        quarter = max(1, len(values) // 4)
        return float(values[-quarter:].mean() - values[:quarter].mean())

    rises = [rise(s) for s in spans]

    # The rules:
    #   drop       the most intense sections: intensity >= DROP_ENERGY
    #   build      just before a drop, with its energy rising by at least
    #              BUILD_RISE_DB from its first bars to its last (a quiet section that
    #              merely precedes a drop isn't a build: nothing is building)
    #   intro      the first section;  outro: the last one (unless they're drops)
    #   breakdown  low intensity (<= BREAKDOWN_ENERGY) somewhere in the middle
    #   groove     anything else: a mid-energy section (a verse, in a pop song)
    count = len(spans)
    labels = list(given) if given else [None] * count
    if not given:
        for i in range(count):
            if not flat and intensity[i] >= DROP_ENERGY:
                labels[i] = "drop"
    for i in range(count - 1):  # allin1 has no "build", so this rule applies to its labels too
        if labels[i] in (None, "groove") and labels[i + 1] == "drop" and rises[i] >= BUILD_RISE_DB:
            labels[i] = "build"
    if not given:
        if labels[0] is None:
            labels[0] = "intro"
        if count > 1 and labels[-1] is None:
            labels[-1] = "outro"
        for i in range(count):
            if labels[i] is None:
                labels[i] = "breakdown" if not flat and intensity[i] <= BREAKDOWN_ENERGY else "groove"

    # Number each kind in order ("drop 1", "drop 2"). Neighbours with the same label
    # stay separate sections: the boundary between them is still a change in the
    # music, and a possible cue point.
    vocals = _section_vocals(features, spans)
    totals = {label: labels.count(label) for label in LABELS}
    seen: dict[str, int] = {}
    result = []
    for i, label in enumerate(labels):
        seen[label] = seen.get(label, 0) + 1
        result.append(Section(
            start=float(edges[i]), end=float(edges[i + 1]), label=label,
            number=seen[label] if totals.get(label, 0) > 1 else 0,
            energy=float(intensity[i]), bass=float(b[i]), rise_db=rises[i],
            vocals=None if vocals is None else vocals[i],
            chroma=features.chroma[spans[i]].mean(axis=0),
        ))
    return result


def _snap(time: float, grid: np.ndarray) -> float:
    return float(grid[np.argmin(np.abs(grid - time))]) if len(grid) else time


def _nearest_beats(beat_times: np.ndarray, times: np.ndarray) -> np.ndarray:
    index = np.clip(np.searchsorted(beat_times, times), 1, len(beat_times) - 1)
    left_closer = np.abs(times - beat_times[index - 1]) < np.abs(times - beat_times[index])
    return index - left_closer


# Vocals (demucs)
# ===============
# A song's vocals clash with another song's vocals far more than its drums or synths
# do. demucs is a neural network that splits a recording into stems (vocals, drums,
# bass, other); we keep the vocal stem and measure how loud it is at each beat.

_warned_no_demucs = False


def demucs_available() -> bool:
    return importlib.util.find_spec("demucs") is not None


def vocal_levels(path: str | Path, beats: np.ndarray) -> np.ndarray | None:
    """Vocal loudness (dB) at each beat, from a demucs vocal stem (cached). None without demucs."""
    global _warned_no_demucs
    if not demucs_available():
        if not _warned_no_demucs:
            print("  note: demucs isn't installed, so vocal clashes aren't scored (see README)", file=sys.stderr)
            _warned_no_demucs = True
        return None
    stem = _vocal_stem(Path(path))
    if stem is None:
        return None
    vocals, sr = sf.read(stem, dtype="float32")
    rms = librosa.feature.rms(y=vocals, hop_length=FEATURE_HOP)[0]
    frames = librosa.time_to_frames(beats, sr=sr, hop_length=FEATURE_HOP)
    return 20 * np.log10(np.maximum(beat_means(rms, frames)[0], 1e-6))


def stems_dir() -> Path:
    """Where demucs vocal stems are kept: a "stems" folder next to the analysis cache."""
    return analysis.DEFAULT_CACHE_PATH.parent / "stems"


def _vocal_stem(path: Path) -> Path | None:
    """Separate the vocals once, and keep them (mono, 22.05 kHz FLAC) in stems_dir()."""
    stat = path.stat()
    key = hashlib.sha1(f"{path.resolve()}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]
    stem = stems_dir() / f"{path.stem[:40]}-{key}.vocals.flac"
    if stem.exists():
        return stem
    work = stems_dir() / f"tmp-{key}"
    print(f"  separating vocals from {path.name} with demucs (once; this takes a while)...")
    try:
        subprocess.run([sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs",
                        "-o", str(work), str(path)], check=True, capture_output=True, text=True)
        separated = next(work.rglob("vocals.wav"))
        vocals, sr = sf.read(separated, dtype="float32")
        mono = vocals.mean(axis=1) if vocals.ndim == 2 else vocals
        sf.write(stem, librosa.resample(mono, orig_sr=sr, target_sr=ANALYSIS_SR), ANALYSIS_SR)
        return stem
    except (subprocess.CalledProcessError, StopIteration, OSError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        print(f"  warning: demucs failed on {path.name}: {detail.strip()[-300:]}", file=sys.stderr)
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _section_vocals(features: StructureFeatures, spans) -> list[float] | None:
    """0..1 per section: how far up front the vocals are, relative to the whole mix.

    A vocal stem within ~6 dB of the full mix is a lead vocal (1.0); 25 dB or more
    below it is bleed or silence (0.0).
    """
    if features.vocal_db is None:
        return None
    relative = features.vocal_db - features.energy_db
    return [float(np.clip((np.mean(relative[s]) + 25) / 19, 0.0, 1.0)) for s in spans]


# allin1 (optional)
# =================
# allin1 is a neural network that finds sections AND names them (intro, verse,
# chorus, ...). If it's installed we use its boundaries (still snapped to our phrase
# grid) and map its names onto ours.

ALLIN1_LABELS = {"intro": "intro", "outro": "outro", "chorus": "drop", "verse": "groove",
                 "bridge": "breakdown", "break": "breakdown", "inst": "groove", "solo": "groove"}


def allin1_available() -> bool:
    return importlib.util.find_spec("allin1") is not None


def _allin1_segments(path: str | Path) -> tuple | None:
    if not allin1_available():
        return None
    try:
        import tempfile

        import allin1

        with tempfile.TemporaryDirectory() as work:  # it writes stems and spectrograms as it goes
            result = allin1.analyze(str(path), demix_dir=f"{work}/demix", spec_dir=f"{work}/spec")
        segments = tuple((float(s.start), float(s.end), str(s.label)) for s in result.segments
                         if s.label not in ("start", "end"))
        return segments or None
    except Exception as exc:  # a broken install shouldn't stop the mix
        print(f"  warning: allin1 failed ({exc}); finding sections ourselves", file=sys.stderr)
        return None


def _allin1_labels(features: StructureFeatures, edges: list[float]) -> list[str]:
    """For each of our sections, the allin1 label covering most of it, in our vocabulary."""
    labels = []
    for a, b in zip(edges, edges[1:]):
        overlap = {}
        for s, e, label in features.segments:
            overlap[label] = overlap.get(label, 0.0) + max(0.0, min(b, e) - max(a, s))
        best = max(overlap, key=overlap.get) if overlap else "verse"
        labels.append(ALLIN1_LABELS.get(best, "groove"))
    return labels
