"""automix: turn a set of songs into one continuous DJ-style mix.

Stage 3: smart ordering and tempo ramps.

  * The set is ordered so each song flows into the next: close tempos, compatible
    keys (the Camelot wheel), no sudden energy jumps (--order smart, the default),
    optionally building up or peaking (--energy rise / peak).
  * Each song plays at its own tempo, and changes tempo only around its
    transitions, gliding to a shared tempo so the two songs' beats line up while
    they overlap (--tempo-mode ramp, the default). Stage 2's one tempo for
    everything is still there as --tempo-mode global.

    python automix.py songs/ -o mix.mp3                     # ordered, beat-matched mix
    python automix.py songs/ --plan-only                    # just show the planned set
    python automix.py songs/ -o mix.mp3 --tempo-mode global # every song at one tempo
    python automix.py songs/ --analyze-only                 # each song's tempo, key, energy
    python automix.py songs/ -o check.wav --clicks          # hear the detected beats
    python automix.py songs/ -o mix.wav --no-beatmatch      # Stage 1: plain crossfades

The pipeline, and where each step lives:

    for each song:  load, resample to 44.1 kHz, stereo   audio_io.load_audio
                    find tempo, beats, key and energy    analysis.detect_beats, estimate_key,
                      (or reuse them from the cache)       energy_features, AnalysisCache
                    trim silence at both ends            processing.trim_silence
                    match loudness (RMS)                 processing.normalize_loudness
    then:           fix half/double tempo readings       analysis.octave_factors
                    rate energy within the set           analysis.energy_scores
                    choose the order                     ordering.order_songs
      ramp mode:    plan tempo ramps and overlaps        mixer.plan_ramped (+ tempo.transition_tempo)
                    stretch just the ramps and overlaps  tempo.apply_time_map
      global mode:  choose the mix tempo                 tempo.target_bpm
                    time-stretch songs to it             tempo.time_stretch
                    plan beat-aligned crossfades         mixer.plan_beatmatched
    then:           crossfade the songs together         mixer.mix_planned (+ transitions.py)
                    keep peaks below -1 dBFS             processing.clipping_gain
                    write the file and set_plan.json     audio_io.save_audio
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

import cuedebug
from analysis import (
    AnalysisCache,
    BeatAnalysis,
    EnergyFeatures,
    SongAnalysis,
    apply_octave,
    beat_this_available,
    beat_this_beats,
    camelot,
    detect_beats,
    energy_features,
    energy_scores,
    estimate_key,
    find_downbeats,
    fit_beat_grid,
    key_method,
    octave_factors,
)
from audio_io import INPUT_EXTENSIONS, SAMPLE_RATE, check_output_path, load_audio, save_audio
from cues import (
    TRANSITION_KINDS,
    CueOption,
    CueSettings,
    CueSong,
    best_options,
    blend_beats,
    choose_cues,
    find_cues,
)
from mixer import (
    CuedPlan,
    CuedTransition,
    MixPlan,
    RampPlan,
    mix_planned,
    plan_beatmatched,
    plan_cued,
    plan_fixed,
    plan_ramped,
)
from ordering import (
    Ordering,
    TempoSettings,
    Track,
    TransitionCost,
    cost_matrices,
    order_songs,
    transition_cost,
)
from processing import CEILING_DB, TARGET_RMS_DB, non_silent_bounds, normalize_loudness, rms_db
from structure import Structure, StructureFeatures, analyze_structure, demucs_available, sections
from tempo import (
    MAX_STRETCH,
    MEET_MODES,
    NO_STRETCH_BELOW,
    apply_time_map,
    ramp_stretch_available,
    rubberband_available,
    stretch_beat_times,
    stretch_method,
    stretch_rate,
    target_bpm,
    time_stretch,
)
from transitions import CLICK_FADE, echo_tail

TEMPO_MODES = ("ramp", "global")
ORDERS = ("smart", "given", "alpha")
ENERGY_SHAPES = ("any", "rise", "peak")
PLAN_FILE = "set_plan.json"  # written next to the output file
UNSURE_KEY = 0.3  # key confidence below this is shown with a "?" in the tracklist
TRANSITION_CHOICES = ("auto", *TRANSITION_KINDS)
KIND_NAMES = {"blend": "blend", "dropswap": "drop swap", "echo": "echo out", "crossfade": "crossfade"}


@dataclass
class Song:
    """One song on its way through the pipeline."""

    path: Path
    audio: np.ndarray | None = None  # trimmed, normalized (and stretched) audio
    length: int = 0  # samples in `audio` (kept even when the audio isn't, for --plan-only)
    bpm: float | None = None  # the song's own tempo, after any half/double fix
    detected_bpm: float | None = None  # what beat detection read, before that fix
    beats: np.ndarray = field(default_factory=lambda: np.zeros(0))  # beat times (s) in `audio`
    rate: float = 1.0  # global mode's time-stretch rate: 1.03 = played 3% faster
    matched: bool = False  # global mode: stretched to the mix tempo?
    cached: bool = False  # analysis reused from the cache?
    key: str | None = None  # e.g. "A minor"
    key_confidence: float = 0.0
    features: EnergyFeatures | None = None  # loudness, onset rate, brightness
    energy: float | None = None  # 0..1, relative to the other songs in this set
    offset: float = 0.0  # seconds of silence trimmed off the start: file time = audio time + offset
    structure_features: StructureFeatures | None = None  # per-beat structure measurements (file time)
    structure: Structure | None = None  # bars, phrases and sections (file time)

    @property
    def title(self) -> str:
        return self.path.stem

    @property
    def camelot(self) -> str | None:
        return camelot(self.key)

    def track(self) -> Track:
        """What ordering.py needs to know about this song."""
        return Track(self.title, self.bpm, self.key, self.key_confidence, self.energy)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn a set of songs into one continuous, beat-matched DJ-style mix."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="folders of songs and/or audio files (.mp3, .wav, .flac); see --order",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="mix.wav",
        help="file to write: .wav, .flac or .mp3 (MP3 needs ffmpeg). Default: mix.wav. "
        f"The set plan is saved next to it as {PLAN_FILE}",
    )
    parser.add_argument(
        "--order",
        choices=ORDERS,
        default="smart",
        help="smart (default): the order that flows best (tempo, key, energy); given: "
        "files in the order listed, each folder alphabetically; alpha: alphabetical by file name",
    )
    parser.add_argument(
        "--start",
        metavar="FILENAME",
        help="the song that opens the set (its file name, with or without the extension)",
    )
    parser.add_argument(
        "--energy",
        choices=ENERGY_SHAPES,
        default="any",
        help="smart order's energy shape. any (default): just the smoothest transitions; "
        "rise: build up through the set; peak: warm up, peak 2/3 of the way, cool down",
    )
    parser.add_argument(
        "--transitions",
        choices=TRANSITION_CHOICES,
        default="auto",
        help="auto (default): the best technique for each pair of songs (never the same one "
        "3 times in a row); blend: long overlaps with a bass swap; dropswap: cut from one "
        "song's build into the next one's drop; echo: echo the outgoing song out",
    )
    parser.add_argument(
        "--min-play",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="each song plays for at least this long (default: 60); shorter songs play whole",
    )
    parser.add_argument(
        "--phrase-bars",
        type=int,
        default=8,
        metavar="BARS",
        help="bars per phrase (default: 8, i.e. 32 beats); transitions start and end on phrases",
    )
    parser.add_argument(
        "--cue-debug",
        action="store_true",
        help="save each song's sections and cue points as an Audacity label file and a PNG "
        "timeline, in a cue_debug folder next to the output",
    )
    parser.add_argument(
        "--crossfade-beats",
        type=int,
        default=16,
        metavar="BEATS",
        help="length of beat-matched crossfades, in beats (default: 16; mid-song blends round "
        "it up to whole phrases); shortened automatically next to very short songs",
    )
    parser.add_argument(
        "--tempo-mode",
        choices=TEMPO_MODES,
        help="ramp (default): each song keeps its own tempo and changes it only around "
        "transitions; global: every song is stretched to one tempo for its whole length "
        "(the default when --bpm is given)",
    )
    parser.add_argument(
        "--ramp-beats",
        type=int,
        default=16,
        metavar="BEATS",
        help="ramp mode: beats over which a song glides to the transition tempo before "
        "an overlap, and back to its own tempo after one (default: 16)",
    )
    parser.add_argument(
        "--meet",
        choices=MEET_MODES,
        default="middle",
        help="ramp mode: the transition tempo. middle (default): both songs meet halfway, "
        "each changing half as much; incoming: the outgoing song matches the incoming one",
    )
    parser.add_argument(
        "--bpm",
        type=float,
        metavar="BPM",
        help="tempo for the whole mix, with --tempo-mode global (default: the median of "
        "the songs' tempos)",
    )
    parser.add_argument(
        "--max-stretch",
        type=float,
        default=MAX_STRETCH * 100,
        metavar="PERCENT",
        help="largest tempo change allowed, in %% (default: 8); transitions that would "
        "need more (in global mode: songs) get plain crossfades instead",
    )
    parser.add_argument(
        "--crossfade",
        type=float,
        default=8.0,
        metavar="SECONDS",
        help="length of plain crossfades in seconds (default: 8): used with "
        "--no-beatmatch, and wherever two songs can't be beat-matched",
    )
    parser.add_argument(
        "--no-beatmatch",
        action="store_true",
        help="Stage 1 behaviour: keep every song's tempo and use plain crossfades",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=f"print the planned set (order, transitions, timings), save {PLAN_FILE}, "
        "and exit without rendering any audio",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="print each song's tempo, beat count, key and energy, then exit without mixing",
    )
    parser.add_argument(
        "--clicks",
        action="store_true",
        help="put a click on every detected beat, so you can check beat detection by ear",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.crossfade) or args.crossfade < 0:
        parser.error("--crossfade must be a number of seconds, 0 or more")
    if args.crossfade_beats < 1:
        parser.error("--crossfade-beats must be at least 1")
    if args.bpm is not None and (not math.isfinite(args.bpm) or args.bpm <= 0):
        parser.error("--bpm must be a positive number")
    if not math.isfinite(args.max_stretch) or args.max_stretch < 0:
        parser.error("--max-stretch must be a percentage, 0 or more")
    if args.ramp_beats < 0:
        parser.error("--ramp-beats must be 0 or more")
    if not math.isfinite(args.min_play) or args.min_play < 0:
        parser.error("--min-play must be a number of seconds, 0 or more")
    if args.phrase_bars < 1:
        parser.error("--phrase-bars must be at least 1")
    if args.tempo_mode == "ramp" and args.bpm is not None:
        parser.error("--bpm sets one tempo for the whole mix, so it needs --tempo-mode global")
    if args.tempo_mode is None:  # --bpm means one tempo for everything: that's global mode
        args.tempo_mode = "global" if args.bpm is not None else "ramp"
    return args


def collect_input_files(inputs: list[str], exclude: Path | None = None) -> list[Path]:
    """Turn the command-line inputs into an ordered list of audio files.

    A folder contributes its audio files in alphabetical order, ignoring case and
    skipping hidden files (like the "._song.mp3" files macOS leaves on USB drives).
    Files named directly are kept in the order given. `exclude` is dropped from the
    list: that's the output file, so re-running into the songs folder doesn't mix
    the previous mix back in.
    """
    files: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            found = [
                p
                for p in path.iterdir()
                if p.is_file()
                and p.suffix.lower() in INPUT_EXTENSIONS
                and not p.name.startswith(".")
            ]
            files += sorted(found, key=lambda p: (p.name.casefold(), p.name))
        elif path.is_file():
            if path.suffix.lower() not in INPUT_EXTENSIONS:
                raise ValueError(
                    f"{path} isn't a supported audio file (use {', '.join(INPUT_EXTENSIONS)})"
                )
            files.append(path)
        else:
            raise FileNotFoundError(f"no such file or folder: {path}")

    if exclude is not None:
        files = [f for f in files if f.resolve() != exclude.resolve()]
    return files


def find_start(paths: list[Path], name: str) -> Path:
    """The file --start names: matched on its file name, with or without the extension."""
    wanted = Path(name).name.casefold()
    matches = [p for p in paths if wanted in (p.name.casefold(), p.stem.casefold())]
    if not matches:
        raise ValueError(f"--start: no song called {name!r} among the inputs")
    if len(matches) > 1:
        raise ValueError(f"--start: {name!r} matches {len(matches)} songs; give the full file name")
    return matches[0]


def format_timestamp(seconds: float) -> str:
    """0 -> '00:00', 225.9 -> '03:45', 3725 -> '1:02:05'."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


# Analysis
# --------


def analyze(
    path: Path, cache: AnalysisCache, audio: np.ndarray | None = None
) -> tuple[SongAnalysis, bool]:
    """A song's tempo, beats, key, energy and structure, with times measured from the start of the file.

    Reuses the cached result if the file hasn't changed, and it was analyzed the way
    we'd analyze it now (an optional tool installed since, like essentia, beat_this
    or demucs, means the song is analyzed again). Otherwise analyzes the song
    (loading the audio first if it isn't given) and caches the result. Returns
    (analysis, came_from_cache).
    """
    cached = cache.get(path)
    if cached is not None and not _outdated(cached):
        return cached, True
    audio = load_audio(path) if audio is None else audio
    beats = detect_beats(audio)
    beat_times, beat_method, downbeats = beats.beats, "ours", None
    if beats.bpm is not None and beat_this_available():
        found = beat_this_beats(audio)
        if found is not None and len(found[0]) >= 8:
            beat_times, downbeats = found
            beat_method = "beat_this"
    bpm = beats.bpm
    if beat_method == "beat_this":
        bpm = 60.0 / fit_beat_grid(beat_times)[1]
    if bpm is not None and downbeats is None:
        downbeats = find_downbeats(audio, beat_times)
    key = estimate_key(audio)
    structure = analyze_structure(audio, beat_times, downbeats, path=path) if bpm is not None else None
    analysis = SongAnalysis(bpm, beat_times, key=key.key, key_confidence=key.confidence,
                            key_method=key.method, energy=energy_features(audio), downbeats=downbeats,
                            structure=structure, beat_method=beat_method)
    cache.put(path, analysis)
    return analysis, False


def _outdated(cached: SongAnalysis) -> bool:
    """Was this cached analysis made without an optional tool that's installed now?"""
    if cached.key_method is None or (cached.key_method == "librosa" and key_method() == "essentia"):
        return True
    if cached.bpm is not None and cached.beat_method == "ours" and beat_this_available():
        return True
    structure = cached.structure
    return structure is not None and structure.vocal_db is None and demucs_available()


def prepare_songs(paths: list[Path], cache: AnalysisCache | None, sr: int = SAMPLE_RATE,
                  keep_audio: bool = True, show_tempo: bool = True) -> list[Song]:
    """Load, trim and loudness-normalize each song, and analyze it if `cache` is given.

    Files that can't be read, or that turn out to be completely silent, are
    skipped with a message rather than stopping the whole mix. With keep_audio=False
    (--plan-only) only each song's length and analysis are kept, not its audio.
    """
    songs = []
    width = min(40, max((len(p.name) for p in paths), default=0))  # line the columns up
    for number, path in enumerate(paths, start=1):
        counter = f"  [{number}/{len(paths)}]"
        try:
            audio = load_audio(path, sr)
        except Exception as exc:  # corrupt or unreadable file: skip it and carry on
            print(f"{counter} {path.name}: skipped, couldn't decode it as audio ({exc})",
                  file=sys.stderr)
            continue
        # Analysis runs on the untrimmed audio, so cached beat times don't depend on
        # the silence-trimming settings.
        analysis, cached = analyze(path, cache, audio) if cache is not None else (None, False)

        start, end = non_silent_bounds(audio)  # trim first, so silence doesn't skew the loudness
        if end == start:
            print(f"{counter} {path.name}: skipped, it's completely silent", file=sys.stderr)
            continue
        audio = audio[start:end]
        level = rms_db(audio)
        song = Song(path, audio=normalize_loudness(audio) if keep_audio else None, length=len(audio),
                    cached=cached, offset=start / sr)

        details = ""
        if analysis is not None:
            # Beat times count from the start of the file; trimming moved the start.
            beats = analysis.beats - start / sr
            song.beats = beats[(beats >= 0) & (beats < len(audio) / sr)]
            song.bpm = song.detected_bpm = analysis.bpm
            song.key, song.key_confidence, song.features = analysis.key, analysis.key_confidence, analysis.energy
            song.structure_features = analysis.structure
            if show_tempo:
                details += f"  {analysis.bpm:6.1f} BPM" if analysis.bpm else "  no steady beat"
            details += f"  key {key_text(song)}"
        print(f"{counter} {path.name:<{width}}  {format_timestamp(len(audio) / sr)}  "
              f"RMS {level:6.1f} dBFS  gain {TARGET_RMS_DB - level:+5.1f} dB{details}")
        songs.append(song)
    return songs


def analyze_songs(paths: list[Path], cache: AnalysisCache) -> list[Song]:
    """For --analyze-only: each song's analysis, without preparing it for mixing."""
    songs = []
    for number, path in enumerate(paths, start=1):
        try:
            analysis, cached = analyze(path, cache)
        except Exception as exc:  # corrupt or unreadable file: skip it and carry on
            print(f"  [{number}/{len(paths)}] {path.name}: skipped, couldn't analyze it ({exc})",
                  file=sys.stderr)
            continue
        print(f"  [{number}/{len(paths)}] {path.name}: {'reused from the cache' if cached else 'analyzed'}")
        songs.append(Song(path, bpm=analysis.bpm, detected_bpm=analysis.bpm, beats=analysis.beats,
                          cached=cached, key=analysis.key, key_confidence=analysis.key_confidence,
                          features=analysis.energy))
    return songs


def rate_energy(songs: list[Song]) -> None:
    """Give each song its 0..1 energy score, relative to the rest of this set."""
    for song, score in zip(songs, energy_scores([song.features for song in songs])):
        song.energy = score


def key_text(song: Song) -> str:
    """The song's Camelot key for tables: "8A", or "8A?" if detection was unsure, "-" if none."""
    if song.camelot is None:
        return "-"
    return song.camelot + ("?" if song.key_confidence < UNSURE_KEY else "")


# Tempo
# -----


def fix_octaves(songs: list[Song], reference: float | None = None) -> None:
    """Fix half/double tempo readings (a "64 BPM" song among 128s), updating bpm and beats."""
    for song, factor in zip(songs, octave_factors([s.bpm for s in songs], reference=reference)):
        if factor != 1.0:
            fixed = apply_octave(BeatAnalysis(song.bpm, song.beats), factor)
            song.bpm, song.beats = fixed.bpm, fixed.beats


def match_tempos(songs: list[Song], target: float | None, max_stretch: float) -> float | None:
    """Global mode: fix half/double readings, choose the mix tempo, decide each song's stretch.

    Updates each song's bpm, beats, rate and matched. `target` is --bpm (None means
    use the median). Returns the mix tempo, or None if no song has a steady beat.
    """
    fix_octaves(songs, reference=target)
    if target is None:
        target = target_bpm([song.bpm for song in songs])
    if target is None:
        return None
    for song in songs:
        song.rate, song.matched = 1.0, False
        if song.bpm is not None and abs(stretch_rate(song.bpm, target) - 1) <= max_stretch:
            rate = stretch_rate(song.bpm, target)
            song.rate = 1.0 if abs(rate - 1) < NO_STRETCH_BELOW else rate  # close enough already
            song.matched = True
    return target


def report_octave_fixes(songs: list[Song]) -> None:
    for song in songs:
        if song.bpm is not None and song.detected_bpm is not None and song.bpm != song.detected_bpm:
            how = "doubled" if song.bpm > song.detected_bpm else "halved"
            print(f"  note: {song.title} read as {song.detected_bpm:.1f} BPM; {how} to "
                  f"{song.bpm:.1f} to match the other songs")


def report_tempos(songs: list[Song], target: float, from_bpm_option: bool, max_stretch: float) -> None:
    """Global mode: print the mix tempo, half/double fixes, and songs too far off to stretch."""
    source = "set with --bpm" if from_bpm_option else "the median of the songs"
    print(f"\nTempo: mixing at {target:.1f} BPM ({source}), stretching with {stretch_method()}")
    if not rubberband_available():
        print("  (install Rubber Band for better-sounding stretches; see README.md)")
    report_octave_fixes(songs)
    for song in songs:
        if song.bpm is not None and not song.matched:
            needed = (stretch_rate(song.bpm, target) - 1) * 100
            print(f"  warning: {song.title} is {song.bpm:.1f} BPM and would need a {needed:+.1f}% "
                  f"stretch (limit {max_stretch * 100:g}%): keeping its own tempo, with plain "
                  f"crossfades on either side", file=sys.stderr)


def stretch_songs(songs: list[Song], sr: int = SAMPLE_RATE, audio: bool = True) -> None:
    """Global mode: time-stretch every tempo-matched song, moving its beat times along with it.

    With audio=False (--plan-only) only the lengths and beat times are worked out.
    """
    width = min(40, max(len(song.title) for song in songs))
    for song in songs:
        if not song.matched:
            continue
        if audio:
            print(f"  {song.title:<{width}}  {song.bpm:6.1f} BPM  stretch {(song.rate - 1) * 100:+5.1f}%")
        if song.rate != 1.0:
            if audio:
                # Normalize again afterwards: the phase vocoder can lose a little energy
                # (up to ~3 dB on noisy sounds), and every song should enter the mix at
                # the same loudness.
                song.audio = normalize_loudness(time_stretch(song.audio, song.rate, sr))
                song.length = len(song.audio)
            else:
                song.length = round(song.length / song.rate)
            beats = stretch_beat_times(song.beats, song.rate)
            song.beats = beats[beats < song.length / sr]


def global_plain_reason(a: Song, b: Song) -> str:
    unmatched = [s for s in (a, b) if not s.matched]
    if not unmatched:
        return "a song is too short to line up the beats"
    if unmatched[0].bpm is None:
        return f"no steady beat in {unmatched[0].title}"
    return f"{unmatched[0].title} isn't tempo-matched"


def report_transitions(songs: list[Song], plan: MixPlan, crossfade_beats: int,
                       beat_period: float, sr: int = SAMPLE_RATE) -> None:
    """Global mode: print how each pair of songs is joined (beat-matched or plain), and why."""
    for i, (a, b) in enumerate(zip(songs, songs[1:])):
        seconds = plan.crossfades[i] / sr
        if plan.beat_aligned[i]:
            n_beats = round(plan.crossfades[i] / beat_period)
            short = "" if n_beats >= crossfade_beats else " (shortened: song too short)"
            print(f"  {a.title} -> {b.title}: beat-matched, {n_beats} beats ({seconds:.1f} s){short}")
        else:
            print(f"  {a.title} -> {b.title}: plain {seconds:.1f} s crossfade ({global_plain_reason(a, b)})")


def plan_ramp_mix(songs: list[Song], crossfade_beats: int, ramp_beats: int, meet: str,
                  max_stretch: float, fallback_crossfade: int, sr: int = SAMPLE_RATE) -> RampPlan:
    """Ramp mode: plan every transition's meeting tempo, overlap and ramps."""
    return plan_ramped(
        lengths=[song.length for song in songs],
        beats=[song.beats * sr for song in songs],
        bpms=[song.bpm for song in songs],
        crossfade_beats=crossfade_beats,
        ramp_beats=ramp_beats,
        meet=meet,
        max_stretch=max_stretch,
        fallback_crossfade=fallback_crossfade,
        sr=sr,
    )


def plain_reason(a: Song, b: Song, reason: str, stretch: float | None, max_stretch: float) -> str:
    """Why a ramp-mode transition is a plain crossfade, in words."""
    if reason == "no-beat":
        return f"no steady beat in {(a if a.bpm is None else b).title}"
    if reason == "stretch":
        return f"needs a {stretch * 100:.1f}% tempo change; limit {max_stretch * 100:g}%"
    return "a song is too short to line up the beats"


def report_ramp_transitions(songs: list[Song], plan: RampPlan, crossfade_beats: int, ramp_beats: int,
                            max_stretch: float, sr: int = SAMPLE_RATE) -> None:
    """Print how each pair is joined in ramp mode, and warn about any that fell back."""
    crossfades = plan.mix_plan().crossfades
    for a, b, move, seconds in zip(songs, songs[1:], plan.transitions, crossfades):
        if move.aligned:
            t = move.tempo
            short = "" if move.n_beats >= crossfade_beats else " (shortened: song too short)"
            print(f"  {a.title} -> {b.title}: beat-matched at {t.bpm:.1f} BPM, {move.n_beats} beats "
                  f"({seconds / sr:.1f} s){short}; {a.title} {(t.rate_out - 1) * 100:+.1f}%, "
                  f"{b.title} {(t.rate_in - 1) * 100:+.1f}%")
            continue
        why = plain_reason(a, b, move.reason, move.tempo.stretch if move.tempo else None, max_stretch)
        print(f"  {a.title} -> {b.title}: plain {seconds / sr:.1f} s crossfade ({why})")
        if move.reason == "stretch":
            print(f"  warning: {a.title} ({a.bpm:.1f} BPM) -> {b.title} ({b.bpm:.1f} BPM) would need a "
                  f"{move.tempo.stretch * 100:.1f}% tempo change (limit {max_stretch * 100:g}%): "
                  f"using a plain crossfade for this transition", file=sys.stderr)
    for song, ramp_song in zip(songs, plan.songs):
        full = ramp_beats * 60.0 / song.bpm * sr if song.bpm else 0
        if any(ramp.end - ramp.start < full - 1 for ramp in ramp_song.ramps):
            print(f"  note: {song.title} is short, so its tempo ramps were shortened to fit")


def render_ramped(songs: list[Song], plan: RampPlan, sr: int = SAMPLE_RATE) -> MixPlan:
    """Ramp mode: stretch each song's ramps and overlaps, cutting it to the part that plays.

    Replaces each song's audio with its played part, and moves its beats (through
    the same time map) to match. Returns the MixPlan for the result.
    """
    width = min(40, max(len(song.title) for song in songs))
    for song, ramp_song in zip(songs, plan.songs):
        time_map = ramp_song.time_map
        played = (round(float(time_map.inverse(ramp_song.out_start))), round(float(time_map.inverse(ramp_song.out_end))))
        stretched = sum(end - start for start, end in time_map.stretched_regions(*played))
        if stretched:
            print(f"  {song.title:<{width}}  stretching {stretched / sr:5.1f} s of {(played[1] - played[0]) / sr:5.1f} s")
        track = apply_time_map(song.audio, time_map, ramp_song.out_start, ramp_song.out_end, sr=sr)
        song.audio = track.audio
        song.length = len(track.audio)
        beats = track.positions(song.beats * sr) / sr
        song.beats = beats[(beats >= 0) & (beats < song.length / sr)]
    return plan.mix_plan([song.length for song in songs])


# Mid-song transitions (Stage 4)
# ------------------------------
# Times: cues.py works in each song's AUDIO time (seconds into the trimmed, and in
# global mode stretched, audio the mixer gets). The user sees FILE time (seconds
# into the original file, which is what Audacity shows), so reports convert back.


def file_time(song: Song, t: float, rate: float | None = None) -> float:
    """Seconds into `song`'s audio (as mixed) -> seconds into its original file."""
    return t * (song.rate if rate is None else rate) + song.offset


def audio_time(song: Song, t: float, rate: float | None = None) -> float:
    """Seconds into the original file -> seconds into the audio as mixed."""
    return (t - song.offset) / (song.rate if rate is None else rate)


def clock(seconds: float) -> str:
    """2:14 (minutes:seconds, as in a tracklist)."""
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def cue_song(song: Song, phrase_bars: int, mix_bpm: float | None, rate: float | None = None,
             sr: int = SAMPLE_RATE) -> CueSong:
    """What cues.py needs to know about `song`, in its audio time.

    `rate` is how much the song's audio (as it is now) has been stretched: song.rate
    by default; 1.0 for a look before global mode's stretching.
    """
    rate = song.rate if rate is None else rate
    duration = song.length / sr
    bpm = song.bpm if mix_bpm is None or not song.matched else mix_bpm  # global mode: the tempo it's now at
    structure = None
    if song.structure_features is not None and bpm:
        song.structure = sections(song.structure_features, phrase_bars, duration=file_time(song, duration, rate))
        shift = np.vectorize(lambda t: audio_time(song, t, rate), otypes=[float])
        structure = Structure(
            downbeats=shift(song.structure.downbeats), phrases=shift(song.structure.phrases),
            beats_per_bar=song.structure.beats_per_bar,
            sections=[replace(s, start=max(0.0, audio_time(song, s.start, rate)),
                              end=min(duration, audio_time(song, s.end, rate)))
                      for s in song.structure.sections],
        )
    # A transition needs audio after its cue, not detected beats: beat tracking trims
    # the faint beats of a quiet outro, but the outro is still there to blend across.
    start = float(song.beats[0]) if len(song.beats) and bpm else 0.0
    return CueSong(song.title, start, duration, bpm, structure, song.key, song.key_confidence)


def has_mid_song_cues(cue: CueSong) -> bool:
    """Can this song be left or entered anywhere but its very end and start?"""
    return any(c.kind != "end" for c in cue.exits) or any(c.kind not in ("start", "intro") for c in cue.entries)


def cued_transitions(songs: list[Song], cues: list[CueSong], chosen: list[CueOption],
                     sr: int = SAMPLE_RATE) -> list[CuedTransition]:
    """The chosen cue options, in each song's audio samples, for mixer.plan_cued."""
    return [CuedTransition(option.kind, option.exit.time * sr, option.entry.time * sr, option.beats, option.tempo)
            for option in chosen]


def report_cued_transitions(songs: list[Song], chosen: list[CueOption], plan: CuedPlan,
                            sr: int = SAMPLE_RATE) -> None:
    """One line per transition: Song A out @ 2:14 (end of drop 2) -> Song B in @ 0:32 (build) | blend | good."""
    for i, (a, b, option) in enumerate(zip(songs, songs[1:], chosen)):
        out_at, in_at = file_time(a, plan.exits[i] / sr), file_time(b, plan.entries[i] / sr)
        if option.kind == "crossfade":
            out_at, in_at = file_time(a, a.length / sr), file_time(b, 0.0)
        print(f"  {a.title} out @ {clock(out_at)} ({option.exit.label}) -> {b.title} in @ {clock(in_at)} "
              f"({option.entry.label}) | {KIND_NAMES[option.kind]} | {option.label}")
        print(f"      {transition_detail(a, b, option, plan, i, sr)}")


def transition_detail(a: Song, b: Song, option: CueOption, plan: CuedPlan, i: int, sr: int = SAMPLE_RATE) -> str:
    cost = f"cost {option.cost:.2f}"
    t = option.tempo
    if option.kind == "blend":
        return (f"{option.beats}-beat blend at {t.bpm:.1f} BPM ({a.title} {(t.rate_out - 1) * 100:+.1f}%, "
                f"{b.title} {(t.rate_in - 1) * 100:+.1f}%), bass swap on beat {option.beats // 2}; {cost}")
    if option.kind == "dropswap":
        return (f"cut on the downbeat at {t.bpm:.1f} BPM ({a.title} {(t.rate_out - 1) * 100:+.1f}%, "
                f"{b.title} {(t.rate_in - 1) * 100:+.1f}%); {cost}")
    if option.kind == "echo":
        return f"{a.title}'s last beat echoes for {option.beats} beats while {b.title} starts; no tempo change; {cost}"
    overlap = plan.mix_plan().crossfades[i] / sr
    return f"plain {overlap:.1f} s crossfade (a song has no steady beat); {cost}"


def render_cued(songs: list[Song], plan: CuedPlan, sr: int = SAMPLE_RATE) -> MixPlan:
    """Stretch each song's ramps and overlaps, cut it to the part that plays, and add echo tails."""
    width = min(40, max(len(song.title) for song in songs))
    for i, (song, ramp_song) in enumerate(zip(songs, plan.songs)):
        time_map = ramp_song.time_map
        played = (round(float(time_map.inverse(ramp_song.out_start))), round(float(time_map.inverse(ramp_song.out_end))))
        stretched = sum(end - start for start, end in time_map.stretched_regions(*played))
        if stretched:
            print(f"  {song.title:<{width}}  stretching {stretched / sr:5.1f} s of {(played[1] - played[0]) / sr:5.1f} s")
        track = apply_time_map(song.audio, time_map, ramp_song.out_start, ramp_song.out_end, sr=sr)
        audio = track.audio
        beats = track.positions(song.beats * sr) / sr
        song.beats = beats[(beats >= 0) & (beats < len(audio) / sr)]
        if i < len(plan.transitions) and plan.transitions[i].kind == "echo":
            # A's dry signal stops at the cue (with a few ms of fade, no click); its last
            # beat then repeats as the echo tail, which B plays over.
            edge = min(len(audio), round(CLICK_FADE * sr))
            audio = audio.copy()
            audio[len(audio) - edge :] *= np.linspace(1.0, 0.0, edge, dtype=np.float32)[:, np.newaxis]
            send = audio[-plan.echo_delays[i] :]
            audio = np.concatenate([audio, echo_tail(send, plan.overlaps[i], sr)])
        song.audio, song.length = audio, len(audio)
    return plan.mix_plan([song.length for song in songs])


# Ordering
# --------


def arrange(songs: list[Song], order: str, energy_shape: str, start: Path | None,
            settings: TempoSettings, pair_costs: np.ndarray | None = None) -> tuple[list[Song], Ordering]:
    """Put the songs in playing order: as given, alphabetical, or smart (ordering.py).

    `pair_costs` (Stage 4: the cost of each pair's best cue points) replaces the
    song-level transition costs. Also returns what the order costs, so a given
    order can be compared with a smart one; Ordering.order indexes `songs`.
    """
    tracks = [song.track() for song in songs]
    first = next((i for i, song in enumerate(songs) if song.path == start), None)
    if order == "smart":
        ordering = order_songs(tracks, settings, energy_shape, start=first, pair_costs=pair_costs)
        return [songs[i] for i in ordering.order], ordering
    indices = list(range(len(songs)))
    if order == "alpha":
        indices.sort(key=lambda i: (songs[i].path.name.casefold(), songs[i].path.name))
    if first is not None:  # --start with given/alpha order: move that song to the front
        indices.remove(first)
        indices.insert(0, first)
    costs, positions = cost_matrices(tracks, settings, energy_shape, pair_costs=pair_costs)
    # The energy shape's positions are by place in the set, so reorder them to match.
    shaped = positions[indices] if len(indices) else positions
    cost = float(costs[indices[:-1], indices[1:]].sum() + shaped[np.arange(len(indices)), np.arange(len(indices))].sum()) \
        if indices else 0.0
    return [songs[i] for i in indices], Ordering(indices, cost, order)


def report_order(songs: list[Song], order: str, energy_shape: str, ordering: Ordering,
                 start: Path | None) -> None:
    opening = f", opening with {songs[0].title}" if start is not None and songs else ""
    shape = "" if energy_shape == "any" else f", energy shape '{energy_shape}'"
    if order != "smart":
        print(f"\nOrder: {'as given' if order == 'given' else 'alphabetical'}{opening}{shape}: "
              f"total cost {ordering.cost:.2f}")
        return
    how = (f"tried all {math.factorial(len(songs) - (start is not None)):,} orders"
           if ordering.method == "brute force" else "greedy nearest neighbour + 2-opt")
    print(f"\nOrder: smart ({how}{shape}{opening}): total cost {ordering.cost:.2f}")


# The set plan: tracklist and set_plan.json
# -----------------------------------------


@dataclass
class TransitionInfo:
    """How one transition turned out, for the tracklist and set_plan.json."""

    beat_matched: bool
    seconds: float  # length of the overlap
    cost: TransitionCost
    tempo: float | None = None  # the tempo both songs play at during the overlap
    stretch_out: float | None = None  # the outgoing song's tempo change there (0.025 = +2.5%)
    stretch_in: float | None = None  # the incoming song's
    beats: int | None = None  # overlap length in beats
    reason: str = ""  # why it's a plain crossfade


def describe_transitions(songs: list[Song], crossfades: list[int], settings: TempoSettings,
                         ramp_plan: RampPlan | None, plan: MixPlan, sr: int = SAMPLE_RATE) -> list[TransitionInfo]:
    infos = []
    for i, (a, b) in enumerate(zip(songs, songs[1:])):
        cost = transition_cost(a.track(), b.track(), settings)
        info = TransitionInfo(plan.beat_aligned[i], crossfades[i] / sr, cost)
        if ramp_plan is not None:
            move = ramp_plan.transitions[i]
            if move.aligned:
                info.tempo, info.beats = move.tempo.bpm, move.n_beats
                info.stretch_out, info.stretch_in = move.tempo.rate_out - 1, move.tempo.rate_in - 1
            else:
                info.reason = plain_reason(a, b, move.reason, move.tempo.stretch if move.tempo else None,
                                           settings.max_stretch)
        elif settings.mode == "global":
            if info.beat_matched:
                info.tempo, info.beats = settings.target, round(crossfades[i] / (60 * sr / settings.target))
                info.stretch_out, info.stretch_in = a.rate - 1, b.rate - 1
            else:
                info.reason = global_plain_reason(a, b)
        infos.append(info)
    return infos


def transition_text(info: TransitionInfo, a: Song, b: Song, show_tempo: bool) -> str:
    verdict = f"cost {info.cost.total:.2f}, {info.cost.label}"
    if info.beat_matched:
        return (f"{info.tempo:.1f} BPM: {a.title} {info.stretch_out * 100:+.1f}%, {b.title} "
                f"{info.stretch_in * 100:+.1f}% ({info.beats} beats)  [{verdict}]")
    if not show_tempo:
        return f"{info.seconds:.1f} s crossfade  [{verdict}]"
    return f"plain {info.seconds:.1f} s crossfade ({info.reason})  [{verdict}]"


def print_tracklist(songs: list[Song], starts: list[int], transitions: list[TransitionInfo], heading: str,
                    show_tempo: bool, show_key: bool, sr: int = SAMPLE_RATE) -> None:
    """Each song's position, start time, title, BPM, key and energy, with each transition between."""
    width = min(40, max(5, max(len(song.title) for song in songs)))
    times = [format_timestamp(start / sr) for start in starts]
    time_width = max(5, max(len(t) for t in times))
    header = f"  {'#':>3}  {'start':<{time_width}}  {'title':<{width}}"
    header += f"  {'BPM':>6}" if show_tempo else ""
    header += f"  {'key':>4}  {'energy':>6}" if show_key else ""
    print(f"\nTracklist ({heading})")
    print(header)
    for i, (song, time) in enumerate(zip(songs, times)):
        row = f"  {i + 1:>2}.  {time:<{time_width}}  {song.title:<{width}}"
        if show_tempo:
            row += f"  {song.bpm:6.1f}" if song.bpm else f"  {'-':>6}"
        if show_key:
            row += f"  {key_text(song):>4}  " + (f"{song.energy:6.2f}" if song.energy is not None else f"{'-':>6}")
        print(row.rstrip())
        if i < len(transitions):
            print(f"        -> {transition_text(transitions[i], song, songs[i + 1], show_tempo)}")


def tracklist_heading(order: str, energy_shape: str, settings: TempoSettings) -> str:
    parts = [f"{order} order" + (f", energy {energy_shape}" if order == "smart" and energy_shape != "any" else "")]
    if settings.mode == "ramp":
        meet = "in the middle" if settings.meet == "middle" else "at the incoming song's tempo"
        parts.append(f"each song at its own tempo, meeting {meet}")
    elif settings.mode == "global":
        parts.append(f"mixed at {settings.target:.1f} BPM")
    else:
        parts.append("plain crossfades")
    return "; ".join(parts)


def set_plan(songs: list[Song], starts: list[int], transitions: list[TransitionInfo], args: argparse.Namespace,
             settings: TempoSettings, total: int, ordering_cost: float, method: str,
             sr: int = SAMPLE_RATE) -> dict:
    """Everything about the planned set, for set_plan.json."""

    def number(value, digits=3):
        return None if value is None else round(float(value), digits)

    return {
        "output": str(args.output),
        "settings": {
            "order": args.order, "start": args.start, "energy": args.energy,
            "tempo_mode": settings.mode, "meet": args.meet if settings.mode == "ramp" else None,
            "mix_bpm": number(settings.target, 2), "crossfade_beats": args.crossfade_beats,
            "ramp_beats": args.ramp_beats if settings.mode == "ramp" else None,
            "max_stretch_percent": args.max_stretch, "plain_crossfade_seconds": args.crossfade,
        },
        "ordering": {"method": method, "total_cost": number(ordering_cost)},
        "total_seconds": number(total / sr, 2),
        "songs": [
            {
                "position": i + 1, "title": song.title, "file": str(song.path),
                "start_seconds": number(start / sr, 2), "start": format_timestamp(start / sr),
                "bpm": number(song.bpm, 2), "detected_bpm": number(song.detected_bpm, 2),
                "key": song.key, "camelot": song.camelot, "key_confidence": number(song.key_confidence, 2),
                "energy": number(song.energy),
            }
            for i, (song, start) in enumerate(zip(songs, starts))
        ],
        "transitions": [
            {
                "from": a.title, "to": b.title,
                "type": "beat-matched" if t.beat_matched else "plain crossfade",
                "tempo_bpm": number(t.tempo, 2),
                "stretch_percent": None if t.stretch_out is None else
                {"from": number(t.stretch_out * 100, 2), "to": number(t.stretch_in * 100, 2)},
                "beats": t.beats, "seconds": number(t.seconds, 2), "reason": t.reason or None,
                "cost": number(t.cost.total), "label": t.cost.label,
                "cost_parts": {"tempo": number(t.cost.tempo), "key": number(t.cost.key),
                               "energy": number(t.cost.energy)},
                "key_relation": t.cost.key_relation,
            }
            for a, b, t in zip(songs, songs[1:], transitions)
        ],
    }


def save_set_plan(plan: dict, output: Path) -> None:
    path = output.parent / PLAN_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, indent=2) + "\n")
        print(f"\nSet plan saved to {path}")
    except OSError as exc:
        print(f"warning: couldn't save the set plan to {path} ({exc})", file=sys.stderr)


def mix_starts(lengths: list[int], crossfades: list[int]) -> list[int]:
    """Where each track starts in the mix (as mixer.render_mix lays them out)."""
    starts = [0]
    for length, overlap in zip(lengths, crossfades):
        starts.append(starts[-1] + length - overlap)
    return starts


def print_analysis_table(songs: list[Song], target: float | None, from_bpm_option: bool,
                         max_stretch: float, tempo_mode: str = "global") -> None:
    """The --analyze-only report: tempo, beats, key and energy (and, in global mode, stretch)."""
    stretches = tempo_mode == "global"
    if not any(song.bpm for song in songs):
        print("\nNo song has a steady beat, so there's nothing to tempo-match.")
    elif stretches:
        source = "set with --bpm" if from_bpm_option else "the median of the songs"
        print(f"\nMix tempo: {target:.1f} BPM ({source}); max stretch {max_stretch * 100:g}%")
    else:
        print(f"\nRamp mode: each song keeps its own tempo, changing it (by at most "
              f"{max_stretch * 100:g}%) only around transitions; --plan-only shows them")
    width = min(40, max(4, max(len(song.title) for song in songs)))
    header = f"\n  {'#':>2}  {'song':<{width}}  {'BPM':>6}  {'beats':>5}"
    header += f"  {'stretch':>7}" if stretches else ""
    print(header + f"  {'key':<12}  {'sure':>4}  {'energy':>6}  note")
    for number, song in enumerate(songs, start=1):
        notes = []
        if song.bpm is not None and song.detected_bpm is not None and song.bpm != song.detected_bpm:
            how = "doubled" if song.bpm > song.detected_bpm else "halved"
            notes.append(f"read as {song.detected_bpm:.1f}, {how}")
        if song.bpm is None:
            bpm, stretch = f"{'-':>6}", f"{'-':>7}"
            notes.append("no steady beat")
        else:
            bpm = f"{song.bpm:6.1f}"
            stretch = f"{(stretch_rate(song.bpm, target) - 1) * 100:+6.1f}%" if target else f"{'-':>7}"
            if stretches and target and not song.matched:
                notes.append(f"over the {max_stretch * 100:g}% limit: won't be stretched")
        key = f"{song.camelot} {song.key}" if song.key else "-"
        energy = f"{song.energy:6.2f}" if song.energy is not None else f"{'-':>6}"
        row = f"  {number:2d}  {song.title:<{width}}  {bpm}  {len(song.beats):5d}"
        row += f"  {stretch}" if stretches else ""
        print(f"{row}  {key:<12}  {song.key_confidence:4.2f}  {energy}  {'; '.join(notes)}".rstrip())
    if stretches:
        print("\n  stretch = tempo change needed to reach the mix tempo (+ = faster)")
    print("  sure = key confidence (0 = a guess, 1 = clear); energy is relative to these songs (0 to 1)")


# The command
# -----------


def played_range(song: Song, ramp_song, sr: int = SAMPLE_RATE) -> tuple[float, float]:
    """The part of `song` that plays: (from, to) in seconds of its original file."""
    time_map = ramp_song.time_map
    return (file_time(song, float(time_map.inverse(ramp_song.out_start)) / sr),
            file_time(song, float(time_map.inverse(ramp_song.out_end)) / sr))


def mix_structured(songs: list[Song], args: argparse.Namespace, settings: TempoSettings, cue_settings: CueSettings,
                   start: Path | None, output: Path, crossfade: int, show_key: bool, sr: int = SAMPLE_RATE) -> int:
    """Stage 4: order the set by its cue points, mix at them, and report (see cues.py and mixer.plan_cued)."""
    if settings.mode == "global":
        report_tempos(songs, settings.target, args.bpm is not None, settings.max_stretch)
        stretch_songs(songs, audio=not args.plan_only)
    else:
        meet = "meet in the middle" if args.meet == "middle" else "go to the incoming song's tempo"
        print(f"\nTempo: ramp mode: each song plays at its own tempo, and around each beat-matched "
              f"transition the songs {meet} (stretching with Rubber Band's R3 engine)")
        report_octave_fixes(songs)

    cue_list = [cue_song(song, args.phrase_bars, settings.target) for song in songs]
    for cue in cue_list:
        find_cues(cue, cue_settings)
    stretches = None
    if settings.mode == "global":
        stretches = [[max(abs(a.rate - 1), abs(b.rate - 1)) for b in songs] for a in songs]
    options = best_options(cue_list, cue_settings, stretches)
    pair_costs = np.zeros((len(songs), len(songs)))
    for (i, j), choices in options.items():
        pair_costs[i, j] = choices[0].cost

    ordered, ordering = arrange(songs, args.order, args.energy, start, settings, pair_costs)
    report_order(ordered, args.order, args.energy, ordering, start)
    index = ordering.order
    cues = [cue_list[i] for i in index]
    chain = [options[index[k], index[k + 1]] for k in range(len(index) - 1)]
    chosen = choose_cues(cues, chain, cue_settings, variety=args.transitions == "auto")
    plan = plan_cued(
        lengths=[song.length for song in ordered],
        beats=[song.beats * sr for song in ordered],
        bpms=[cue.bpm for cue in cues],
        transitions=cued_transitions(ordered, cues, chosen, sr),
        ramp_beats=args.ramp_beats if settings.mode == "ramp" else 0,
        fallback_crossfade=crossfade,
        sr=sr,
        audio=None if args.plan_only else [song.audio for song in ordered],
    )
    print(f"\n{'Planned' if args.plan_only else 'Mixing'} transitions at cue points (--transitions "
          f"{args.transitions}; {args.phrase_bars}-bar phrases; each song plays at least {args.min_play:g} s):")
    report_cued_transitions(ordered, chosen, plan, sr)
    ranges = [played_range(song, ramp_song, sr) for song, ramp_song in zip(ordered, plan.songs)]

    if args.cue_debug:
        cuedebug.write(output.parent / "cue_debug", ordered, cues, chosen, plan, ranges, file_time, sr)

    if args.plan_only:
        mix_plan = plan.mix_plan()
        lengths = plan.lengths()
        starts = mix_starts(lengths, mix_plan.crossfades)
        total = starts[-1] + lengths[-1]
    else:
        mix_plan = render_cued(ordered, plan, sr)
        clicks = [song.beats * sr for song in ordered] if args.clicks else None
        mix = mix_planned([song.audio for song in ordered], mix_plan, plan.transition_functions(), clicks=clicks)
        for song in ordered:
            song.audio = None
        if mix.safety_gain_db < -0.05:
            print(f"  note: whole mix turned down {-mix.safety_gain_db:.1f} dB so its loudest "
                  f"peak stays under {CEILING_DB:g} dBFS (see processing.prevent_clipping)")
        print(f"Writing {output}...")
        save_audio(output, mix.audio)
        starts, total = mix.starts, len(mix.audio)

    print_cued_tracklist(ordered, starts, chosen, plan, ranges, tracklist_heading(args.order, args.energy, settings),
                         show_key, sr)
    print(f"\n{'Planned length' if args.plan_only else 'Total length'}: {format_timestamp(total / sr)}")
    save_set_plan(cued_set_plan(ordered, starts, chosen, plan, ranges, args, settings, total, ordering, sr), output)
    return 0


def print_cued_tracklist(songs: list[Song], starts: list[int], chosen: list[CueOption], plan: CuedPlan,
                         ranges: list[tuple[float, float]], heading: str, show_key: bool, sr: int = SAMPLE_RATE) -> None:
    """The Stage 3 tracklist, plus which part of each song plays, and each transition's cue points and type."""
    width = min(40, max(5, max(len(song.title) for song in songs)))
    times = [format_timestamp(s / sr) for s in starts]
    time_width = max(5, max(len(t) for t in times))
    header = f"  {'#':>3}  {'start':<{time_width}}  {'title':<{width}}  {'BPM':>6}"
    header += f"  {'key':>4}  {'energy':>6}" if show_key else ""
    print(f"\nTracklist ({heading})")
    print(header + "  plays")
    for i, (song, time, (play_from, play_to)) in enumerate(zip(songs, times, ranges)):
        row = f"  {i + 1:>2}.  {time:<{time_width}}  {song.title:<{width}}"
        row += f"  {song.bpm:6.1f}" if song.bpm else f"  {'-':>6}"
        if show_key:
            row += f"  {key_text(song):>4}  " + (f"{song.energy:6.2f}" if song.energy is not None else f"{'-':>6}")
        print(f"{row}  {clock(play_from)}-{clock(play_to)}")
        if i < len(chosen):
            option, nxt = chosen[i], songs[i + 1]
            out_at = file_time(song, plan.exits[i] / sr)
            in_at = file_time(nxt, plan.entries[i] / sr)
            print(f"        -> out @ {clock(out_at)} ({option.exit.label}), {nxt.title} in @ {clock(in_at)} "
                  f"({option.entry.label}) | {KIND_NAMES[option.kind]}  [cost {option.cost:.2f}, {option.label}]")


def cued_set_plan(songs: list[Song], starts: list[int], chosen: list[CueOption], plan: CuedPlan,
                  ranges: list[tuple[float, float]], args: argparse.Namespace, settings: TempoSettings,
                  total: int, ordering: Ordering, sr: int = SAMPLE_RATE) -> dict:
    """set_plan.json for a Stage 4 mix: the Stage 3 fields, plus sections, cue points and transition types."""

    def number(value, digits=3):
        return None if value is None else round(float(value), digits)

    result = {
        "output": str(args.output),
        "settings": {
            "order": args.order, "start": args.start, "energy": args.energy, "tempo_mode": settings.mode,
            "meet": args.meet if settings.mode == "ramp" else None, "mix_bpm": number(settings.target, 2),
            "transitions": args.transitions, "phrase_bars": args.phrase_bars, "min_play_seconds": args.min_play,
            "crossfade_beats": args.crossfade_beats, "ramp_beats": args.ramp_beats if settings.mode == "ramp" else None,
            "max_stretch_percent": args.max_stretch, "plain_crossfade_seconds": args.crossfade,
        },
        "ordering": {"method": ordering.method, "total_cost": number(ordering.cost)},
        "total_seconds": number(total / sr, 2),
        "songs": [],
        "transitions": [],
    }
    for i, (song, start, (play_from, play_to)) in enumerate(zip(songs, starts, ranges)):
        result["songs"].append({
            "position": i + 1, "title": song.title, "file": str(song.path),
            "start_seconds": number(start / sr, 2), "start": format_timestamp(start / sr),
            "plays_from": number(play_from, 2), "plays_to": number(play_to, 2),
            "bpm": number(song.bpm, 2), "detected_bpm": number(song.detected_bpm, 2),
            "key": song.key, "camelot": song.camelot, "key_confidence": number(song.key_confidence, 2),
            "energy": number(song.energy),
            "sections": [{"start": number(s.start, 2), "end": number(s.end, 2), "label": s.name,
                          "energy": number(s.energy, 2), "vocals": number(s.vocals, 2)}
                         for s in (song.structure.sections if song.structure else [])],
        })
    for i, (a, b, option) in enumerate(zip(songs, songs[1:], chosen)):
        t = option.tempo
        result["transitions"].append({
            "from": a.title, "to": b.title, "type": KIND_NAMES[option.kind],
            "out": {"time": number(file_time(a, plan.exits[i] / sr), 2), "cue": option.exit.label},
            "in": {"time": number(file_time(b, plan.entries[i] / sr), 2), "cue": option.entry.label},
            "tempo_bpm": number(t.bpm, 2) if t else None,
            "stretch_percent": {"from": number((t.rate_out - 1) * 100, 2), "to": number((t.rate_in - 1) * 100, 2)}
            if t else None,
            "beats": option.beats or None,
            "bass_swap_beat": option.beats // 2 if option.kind == "blend" else None,
            "seconds": number((plan.overlaps[i] if plan.overlaps[i] is not None
                               else plan.mix_plan().crossfades[i]) / sr, 3),
            "cost": number(option.cost), "label": option.label,
            "cost_parts": {key: number(value) for key, value in option.parts.items()},
        })
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    try:
        if not (args.analyze_only or args.plan_only):
            check_output_path(output)  # fail now, not after minutes of processing
        paths = collect_input_files(args.inputs, exclude=output)
        start = find_start(paths, args.start) if args.start else None
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not paths:
        print(f"error: no audio files ({', '.join(INPUT_EXTENSIONS)}) found in: "
              f"{' '.join(args.inputs)}", file=sys.stderr)
        return 1

    cache = AnalysisCache()
    max_stretch = args.max_stretch / 100

    if args.analyze_only:
        print(f"Analyzing {len(paths)} song(s)...")
        songs = analyze_songs(paths, cache)
        if not songs:
            print("error: none of the files could be used", file=sys.stderr)
            return 1
        rate_energy(songs)
        target = None
        if args.tempo_mode == "global":
            target = match_tempos(songs, args.bpm, max_stretch)
        else:
            fix_octaves(songs)
        print_analysis_table(songs, target, args.bpm is not None, max_stretch, args.tempo_mode)
        return 0

    beatmatch = not args.no_beatmatch
    analyze_songs_too = beatmatch or args.clicks or args.order == "smart"
    print(f"Preparing {len(paths)} song(s): trimming silence, "
          f"normalizing loudness to {TARGET_RMS_DB:g} dBFS RMS"
          + (", finding beats" if beatmatch or args.clicks else "")
          + (", key and energy" if analyze_songs_too else ""))
    songs = prepare_songs(paths, cache if analyze_songs_too else None, keep_audio=not args.plan_only,
                          show_tempo=beatmatch or args.clicks)
    if not songs:
        print("error: none of the files could be used", file=sys.stderr)
        return 1
    if start is not None and all(song.path != start for song in songs):
        print(f"error: --start: {start.name} couldn't be used (see above)", file=sys.stderr)
        return 1
    reused = sum(song.cached for song in songs)
    if analyze_songs_too and reused:
        print(f"  (analysis of {reused} of {len(songs)} song(s) reused from {cache.path.name})")
    rate_energy(songs)

    # Tempo decisions that don't depend on the order come first, because the order
    # depends on them: ordering scores each transition by the stretch it will need.
    crossfade = round(args.crossfade * SAMPLE_RATE)  # seconds -> samples
    tempo_mode = args.tempo_mode if beatmatch else None
    if tempo_mode == "ramp" and not ramp_stretch_available():
        print("warning: --tempo-mode ramp needs Rubber Band 3.0 or newer, which isn't installed "
              "(see README.md): using --tempo-mode global instead", file=sys.stderr)
        tempo_mode = "global"
    if tempo_mode is not None and not any(song.bpm for song in songs):
        print("\nNo steady beat found in any song, so nothing to beat-match.")
        tempo_mode = None
    target = None
    if tempo_mode == "ramp":
        fix_octaves(songs)
    elif tempo_mode == "global":
        target = match_tempos(songs, args.bpm, max_stretch)
    settings = TempoSettings(tempo_mode or "none", args.meet, max_stretch, target)

    # Stage 4: if any song has a structure to cue from (or a transition type was asked
    # for), mix at cue points. Otherwise (short songs, songs with no clear sections)
    # it's the Stage 3 mix: songs whole, joined end to start.
    if tempo_mode is not None:
        kinds = TRANSITION_KINDS if args.transitions == "auto" else (args.transitions,)
        cue_settings = CueSettings(settings, args.phrase_bars, blend_beats(args.crossfade_beats, args.phrase_bars),
                                   args.min_play, kinds)
        probe = [cue_song(song, args.phrase_bars, None, rate=1.0) for song in songs]
        for cue in probe:
            find_cues(cue, cue_settings)
        if args.transitions != "auto" or any(has_mid_song_cues(cue) for cue in probe):
            return mix_structured(songs, args, settings, cue_settings, start, output, crossfade, analyze_songs_too)

    songs, ordering = arrange(songs, args.order, args.energy, start, settings)
    report_order(songs, args.order, args.energy, ordering, start)

    ramp_plan = None
    if tempo_mode == "ramp":
        meet = "meet in the middle" if args.meet == "middle" else "go to the incoming song's tempo"
        print(f"\nTempo: ramp mode: each song plays at its own tempo, and at each transition "
              f"the songs {meet} (stretching with Rubber Band's R3 engine)")
        report_octave_fixes(songs)
        ramp_plan = plan_ramp_mix(songs, args.crossfade_beats, args.ramp_beats, args.meet,
                                  max_stretch, crossfade)
        print(f"\n{'Planned' if args.plan_only else 'Mixing with'} {args.crossfade_beats}-beat "
              f"crossfades and {args.ramp_beats}-beat tempo ramps...")
        report_ramp_transitions(songs, ramp_plan, args.crossfade_beats, args.ramp_beats, max_stretch)
        plan = ramp_plan.mix_plan()
    elif tempo_mode == "global":
        report_tempos(songs, target, args.bpm is not None, max_stretch)
        stretch_songs(songs, audio=not args.plan_only)
        beat_period = 60.0 / target * SAMPLE_RATE  # samples per beat at the mix tempo
        plan = plan_beatmatched(
            lengths=[song.length for song in songs],
            beats=[song.beats * SAMPLE_RATE for song in songs],
            matched=[song.matched for song in songs],
            beat_period=beat_period,
            crossfade_beats=args.crossfade_beats,
            fallback_crossfade=crossfade,
        )
        print(f"\n{'Planned' if args.plan_only else 'Mixing with'} {args.crossfade_beats}-beat crossfades...")
        report_transitions(songs, plan, args.crossfade_beats, beat_period)
    else:
        plan = plan_fixed([song.length for song in songs], crossfade)
        print(f"\n{'Planned' if args.plan_only else 'Mixing with'} {args.crossfade:g} s crossfades...")
        for i, actual in enumerate(plan.crossfades):
            if actual < crossfade:
                print(f"  note: {songs[i].title} -> {songs[i + 1].title} crossfade shortened to "
                      f"{actual / SAMPLE_RATE:.1f} s (song too short)")

    if args.plan_only:
        lengths = [end - begin for begin, end in plan.cuts]
        starts = mix_starts(lengths, plan.crossfades)
        total = starts[-1] + lengths[-1]
    else:
        if ramp_plan is not None:
            plan = render_ramped(songs, ramp_plan)
        clicks = [song.beats * SAMPLE_RATE for song in songs] if args.clicks else None
        mix = mix_planned([song.audio for song in songs], plan, clicks=clicks)
        for song in songs:
            song.audio = None  # the mix holds its own copy; free this memory before writing the file
        if mix.safety_gain_db < -0.05:
            print(f"  note: whole mix turned down {-mix.safety_gain_db:.1f} dB so its loudest "
                  f"peak stays under {CEILING_DB:g} dBFS (see processing.prevent_clipping)")
        print(f"Writing {output}...")
        save_audio(output, mix.audio)
        starts, total = mix.starts, len(mix.audio)

    transitions = describe_transitions(songs, plan.crossfades, settings, ramp_plan, plan)
    print_tracklist(songs, starts, transitions, tracklist_heading(args.order, args.energy, settings),
                    show_tempo=tempo_mode is not None, show_key=analyze_songs_too)
    print(f"\n{'Planned length' if args.plan_only else 'Total length'}: {format_timestamp(total / SAMPLE_RATE)}")
    save_set_plan(set_plan(songs, starts, transitions, args, settings, total, ordering.cost, ordering.method),
                  output)
    return 0


if __name__ == "__main__":
    # Send each line out as soon as it's printed, even into a pipe or a file, so
    # warnings (stderr) stay in order with the progress messages (stdout).
    sys.stdout.reconfigure(line_buffering=True)
    sys.exit(main())
