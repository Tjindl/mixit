r"""Assemble a list of prepared tracks into one continuous mix.

Building a mix takes two steps:

1. PLAN    decide which part of each track plays and how much neighbours overlap
           -> plan_fixed() (Stage 1: whole tracks, fixed-length crossfades)
           -> plan_beatmatched() (Stage 2: crossfades that line up the beats, for
              songs already stretched to one shared tempo)
           -> plan_ramped() (Stage 3: each song keeps its own tempo, and changes
              it only around transitions; see "Ramp mode" below)
2. RENDER  lay the played parts end to end and blend every overlap with a
           transition -> render_mix()

mix_planned() renders a plan and makes sure the result can't clip. mix_tracks()
is the Stage 1 shortcut. Keeping PLAN and RENDER apart is what let Stages 2 and 3
plug in without touching render_mix().

How the tracks are laid out (3 tracks, 2 crossfades):

    A    |======= A solo =======|-fade out-|
    B                           |-fade in--|===== B solo =====|-fade out-|
    C                                                         |-fade in--|=== C solo ===|

    mix  |======= A solo =======|  A + B   |===== B solo =====|  B + C   |=== C solo ===|
         ^ A starts             ^ B starts                    ^ C starts

Every sample of the mix belongs to exactly one section: either a "solo" section
where one track plays, or a transition where two overlap. So:

    length of mix = sum of (played) track lengths - sum of crossfade lengths

Lining up beats (Stage 2)
-------------------------
Once every song has been time-stretched to the same tempo, beats come at the same
rate in all of them: one every P samples. To line up two songs, pick one beat in
each and make those two beats land on the same sample of the mix:

                    beat:  0      1       2      ...   N
    outgoing A, sample:    a      a+P     a+2P   ...   a+N*P   <- cut A here
    incoming B, sample:    b      b+P     b+2P   ...   b+N*P
    in the mix:            x      x+P     x+2P   ...   x+N*P   <- same sample for both

We cut B's intro off before its mix-in beat b, and cut A off after an N-beat
crossfade that starts at its mix-out beat a. The overlap then plays A[a + k]
together with B[b + k] for every k. So once beat 0 lines up, every following beat
lines up too: A's beat at a + k*P and B's beat at b + k*P land on the same sample.

Ramp mode (Stage 3)
-------------------
Here each song plays at its own tempo, and only the transitions are stretched. For
A (120 BPM) into B (126 BPM), meeting in the middle at 123 BPM:

    A  ==== 120 BPM ====|/ ramp to 123 /|== 123: overlap ==|
    B                                   |== 123: overlap ==|\ ramp to 126 \|== 126 BPM ==
                         <- ramp beats ->  <- crossfade ->   <- ramp beats ->

plan_ramped() finds the mix points the same way as above, but on each song's own
unstretched beats (with that song's own beat period). Then it builds a time map
per song (see tempo.py) that holds the overlap at the transition tempo and ramps
to and from it. Inside an overlap both songs play at the same constant tempo, so
their beats are again exactly P samples apart, and the argument above applies
unchanged: line up beat 0 and every beat after it lines up too.

Mid-song transitions (Stage 4)
------------------------------
plan_cued() does the same for transitions that start anywhere in a song, at cue
points chosen by cues.py, each with its own technique:

    blend     the overlap starts at A's exit cue and B's entry cue (both downbeats
              on phrase boundaries) and lasts a whole phrase; the bass swaps on the
              downbeat halfway through
    drop swap A plays its build right up to the downbeat where its drop would hit,
              and B starts on the downbeat of its drop: a cut, with a few ms of
              crossfade just BEFORE the downbeat, so B's drop lands exactly on it
    echo out  A plays to its exit cue; then its last beat echoes away while B
              starts. Nothing needs to line up, so neither song changes tempo.

Tempo ramps go wherever the beat-synced transitions are: A ramps into the meeting
tempo just before its exit cue, and B ramps back out just after its overlap.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from analysis import octave_beats
from audio_io import SAMPLE_RATE
from processing import CEILING_DB, click_sound, clipping_gain, gain_to_db
from tempo import RAMP_ANCHORS_PER_BEAT, TimeMap, TimeMapBuilder, TransitionTempo, transition_tempo
from transitions import Transition, bass_swap, echo_join, equal_power_crossfade

FIT_BEATS = 16  # extra beats, beyond the crossfade's own, used to pin down the beat grid
CLICK_FREQS = (1500.0, 2500.0)  # --clicks pitch for odd / even songs (see _add_clicks)


@dataclass(frozen=True)
class Mix:
    """A finished mix, plus where everything landed. Positions are in samples."""

    audio: np.ndarray  # float32, shape (n_samples, 2)
    starts: list[int]  # where each track starts (begins fading in)
    crossfades: list[int]  # actual overlap of track i and i+1 (may be less than requested)
    safety_gain_db: float = 0.0  # gain applied to avoid clipping: 0 = untouched, -3 = 3 dB quieter


@dataclass(frozen=True)
class MixPlan:
    """Which part of each track plays, and how neighbours overlap. Positions in samples."""

    cuts: list[tuple[int, int]]  # track i plays from sample cuts[i][0] up to cuts[i][1]
    crossfades: list[int]  # overlap between the played parts of track i and i+1
    beat_aligned: list[bool]  # True where that overlap lines up the two tracks' beats


def plan_fixed(lengths: Sequence[int], crossfade: int) -> MixPlan:
    """Stage 1: play every track whole, with fixed-length crossfades."""
    return MixPlan(
        cuts=[(0, length) for length in lengths],
        crossfades=plan_crossfades(lengths, crossfade),
        beat_aligned=[False] * max(0, len(lengths) - 1),
    )


def plan_beatmatched(
    lengths: Sequence[int],
    beats: Sequence[np.ndarray],
    matched: Sequence[bool],
    beat_period: float,
    crossfade_beats: int,
    fallback_crossfade: int,
) -> MixPlan:
    """Plan a mix whose transitions line up the beats of neighbouring tracks.

    lengths[i]          samples in track i, after time-stretching
    beats[i]            beat positions in track i, in samples, in order
    matched[i]          True if track i was stretched to the shared tempo
    beat_period         samples per beat at the shared tempo
    crossfade_beats     length of a beat-aligned crossfade, in beats
    fallback_crossfade  length, in samples, of the plain Stage 1 crossfade used
                        wherever two neighbours can't be beat-matched (one of them
                        wasn't tempo-matched, or has too few beats)

    A beat-aligned crossfade starts on the incoming track's first beat, and on the
    outgoing track's beat that's `crossfade_beats` before its last one, so the
    crossfade ends on the outgoing track's last beat.
    """
    count = len(lengths)
    cuts = [[0, length] for length in lengths]
    crossfades = [0] * (count - 1)
    aligned = [False] * (count - 1)

    # 1. Beat-aligned transitions come first, because they decide where tracks get cut.
    for i in range(count - 1):
        if not (matched[i] and matched[i + 1]):
            continue
        n_beats = min(crossfade_beats, _beats_to_spare(beats[i]), _beats_to_spare(beats[i + 1]))
        if n_beats < 1:
            continue
        overlap = round(n_beats * beat_period)
        mix_out = _mix_out_beat(beats[i], n_beats, beat_period)
        while mix_out + overlap > lengths[i]:  # if that ran past the end of the track,
            mix_out -= beat_period  # move one whole beat earlier, staying on the grid
        mix_in = _mix_in_beat(beats[i + 1], n_beats, beat_period)
        while mix_in < 0:  # likewise if the grid's first beat lands before the start
            mix_in += beat_period
        fade_in_end = cuts[i][0] + (crossfades[i - 1] if i > 0 and aligned[i - 1] else 0)
        if round(mix_out) < fade_in_end:
            continue  # no room left after track i's fade-in: use a plain crossfade instead
        cuts[i][1] = round(mix_out) + overlap
        cuts[i + 1][0] = round(mix_in)
        crossfades[i] = overlap
        aligned[i] = True

    # 2. Every other transition is a plain Stage 1 crossfade over what's left.
    for i in range(count - 1):
        if not aligned[i]:
            length_a = cuts[i][1] - cuts[i][0]
            length_b = cuts[i + 1][1] - cuts[i + 1][0]
            crossfades[i] = min(fallback_crossfade, length_a // 2, length_b // 2)
    return MixPlan([(start, end) for start, end in cuts], crossfades, aligned)


def _beats_to_spare(beats: np.ndarray) -> int:
    """The longest crossfade, in beats, that a track with these beats can give.

    The same idea as Stage 1's half rule, counted in beats: a track has
    len(beats) - 1 beat intervals, and its fade-in and fade-out may each use at most
    half of them, minus one beat, so at least one beat of solo separates the two.
    (A negative result means the track is too short to beat-match at all.)
    """
    return (len(beats) - 2) // 2


def _grid_offset(beats: np.ndarray, period: float) -> tuple[float, np.ndarray]:
    """Line up a grid of beats `period` samples apart with the detected `beats`.

    Returns (offset, numbers): grid beat k sits at offset + k * period, and numbers[i]
    is the grid beat that detected beat i belongs to. After time-stretching we know
    the spacing exactly, so only the offset is unknown. We take the MEDIAN of the
    beats' offsets from the grid rather than their average: the tracker sometimes
    locks onto an off-beat sound for a few beats at the start or end of a song,
    putting them a 16th note off, and the median lets the majority decide.
    """
    steps = np.maximum(1, np.round(np.diff(beats) / period))  # 2 where a beat was missed
    numbers = np.concatenate([[0.0], np.cumsum(steps)])
    return float(np.median(beats - numbers * period)), numbers


def _mix_out_beat(beats: np.ndarray, n_beats: int, period: float) -> float:
    """Position of the grid beat `n_beats` before the track's last beat.

    Rather than trusting single detected beats, we line up a beat grid with the last
    few dozen beats and read the position off the grid.
    """
    offset, numbers = _grid_offset(beats[-(n_beats + FIT_BEATS) :], period)
    return offset + (numbers[-1] - n_beats) * period


def _mix_in_beat(beats: np.ndarray, n_beats: int, period: float) -> float:
    """Position of the track's first beat, read off a grid lined up with its first beats."""
    offset, _ = _grid_offset(beats[: n_beats + FIT_BEATS], period)
    return offset


# Ramp mode
# ---------


@dataclass(frozen=True)
class Ramp:
    """A smooth tempo change inside one song. Positions are in the song's own samples."""

    start: int
    end: int
    rate_from: float
    rate_to: float


@dataclass(frozen=True)
class RampTransition:
    """How one pair of neighbours is joined in ramp mode."""

    aligned: bool  # beat-matched (True), or a plain crossfade
    overlap: int = 0  # length of the beat-matched overlap, in output samples
    tempo: TransitionTempo | None = None  # where the songs meet (None if either has no beat)
    n_beats: int = 0  # length of the overlap in beats
    reason: str = ""  # why it's plain: "no-beat", "stretch" (beyond --max-stretch) or "short"


@dataclass(frozen=True)
class RampSong:
    """One song's part in a ramp-mode mix."""

    time_map: TimeMap  # where each of its samples ends up (tempo.py)
    out_start: int  # it plays from output sample out_start (in time-map terms)...
    out_end: int  # ...up to out_end
    ramps: tuple[Ramp, ...] = ()  # its tempo ramps, in order

    @property
    def length(self) -> int:
        return self.out_end - self.out_start


@dataclass(frozen=True)
class RampPlan:
    songs: list[RampSong]
    transitions: list[RampTransition]
    fallback_crossfade: int

    def mix_plan(self, lengths: Sequence[int] | None = None) -> MixPlan:
        """The MixPlan for the time-mapped tracks, which are already cut to length.

        `lengths` are the rendered tracks' lengths (they can differ from the planned
        ones by a few samples; see tempo.apply_time_map); plain crossfades are fitted
        to those.
        """
        lengths = list(lengths) if lengths is not None else [song.length for song in self.songs]
        overlaps = [t.overlap if t.aligned else None for t in self.transitions]
        return MixPlan(
            cuts=[(0, length) for length in lengths],
            crossfades=_fit_plain_crossfades(lengths, overlaps, self.fallback_crossfade),
            beat_aligned=[t.aligned for t in self.transitions],
        )


def plan_ramped(
    lengths: Sequence[int],
    beats: Sequence[np.ndarray],
    bpms: Sequence[float | None],
    crossfade_beats: int,
    ramp_beats: int,
    meet: str,
    max_stretch: float,
    fallback_crossfade: int,
    sr: int = SAMPLE_RATE,
) -> RampPlan:
    """Plan a mix where songs keep their own tempo and change it only around transitions.

    lengths[i], beats[i] and bpms[i] describe song i as it is (unstretched): its
    length in samples, its beat positions in samples, and its tempo (None if it has
    no steady beat). Each transition from A to B:

    1. meets at transition_tempo(A, B, meet), unless that needs more than
       `max_stretch` (then it's a plain crossfade instead)
    2. finds the mix points like plan_beatmatched: the overlap of `crossfade_beats`
       beats starts on B's first beat and ends on A's last
    3. ramps A from its own tempo to the meeting tempo over the `ramp_beats` beats
       before the overlap, and B back to its own over the `ramp_beats` after it

    A song too short for both of its ramps gets proportionally shorter ones.
    """
    count = len(lengths)
    periods = [60.0 * sr / bpm if bpm else None for bpm in bpms]
    intro: list[tuple[float, float, float] | None] = [None] * count  # (mix-in, overlap end, rate)
    outro: list[tuple[float, float] | None] = [None] * count  # (mix-out, rate)
    transitions: list[RampTransition] = []

    # 1. Transitions: where each overlap is, and at what tempo.
    for a in range(count - 1):
        b = a + 1
        if periods[a] is None or periods[b] is None:
            transitions.append(RampTransition(False, reason="no-beat"))
            continue
        tempo = transition_tempo(bpms[a], bpms[b], meet)
        if tempo.stretch > max_stretch:
            transitions.append(RampTransition(False, tempo=tempo, reason="stretch"))
            continue
        period_a = periods[a]
        period_b = periods[b] / tempo.factor  # B's beats, counted at the tempo it's matched as
        beats_b = octave_beats(beats[b], tempo.factor)
        n_beats = min(crossfade_beats, _beats_to_spare(beats[a]), _beats_to_spare(beats_b))
        if n_beats < 1:
            transitions.append(RampTransition(False, tempo=tempo, reason="short"))
            continue
        mix_out = _mix_out_beat(beats[a], n_beats, period_a)
        while mix_out + n_beats * period_a > lengths[a]:  # stay inside A, on its beat grid
            mix_out -= period_a
        mix_in = _mix_in_beat(beats_b, n_beats, period_b)
        while mix_in < 0:
            mix_in += period_b
        if intro[a] is not None and mix_out < intro[a][1]:
            # A is too short: its fade-out would start before its fade-in has finished.
            transitions.append(RampTransition(False, tempo=tempo, reason="short"))
            continue
        outro[a] = (mix_out, tempo.rate_out)
        intro[b] = (mix_in, mix_in + n_beats * period_b, tempo.rate_in)
        overlap = round(n_beats * 60.0 * sr / tempo.bpm)  # N beats at the meeting tempo
        transitions.append(RampTransition(True, overlap, tempo, n_beats))

    # 2. Songs: a time map for each, holding every overlap at its meeting tempo and
    #    ramping to and from it.
    songs = []
    for s in range(count):
        rate_in = intro[s][2] if intro[s] else 1.0
        rate_out = outro[s][1] if outro[s] else 1.0
        ramp_length = ramp_beats * periods[s] if periods[s] else 0.0
        len_in = ramp_length if rate_in != 1.0 else 0.0
        len_out = ramp_length if rate_out != 1.0 else 0.0
        room_start = intro[s][1] if intro[s] else 0.0
        room_end = outro[s][0] if outro[s] else float(lengths[s])
        if len_in + len_out > room_end - room_start:  # a short song: shrink both ramps to fit
            scale = max(0.0, room_end - room_start) / (len_in + len_out)
            len_in, len_out = len_in * scale, len_out * scale

        def anchors(length: float) -> int:
            return max(2, round(RAMP_ANCHORS_PER_BEAT * length / periods[s]))

        builder, ramps = TimeMapBuilder(), []
        if intro[s]:
            builder.hold(intro[s][1], rate_in)  # everything up to the overlap's end at its tempo
            if len_in:
                builder.ramp(intro[s][1] + len_in, rate_in, 1.0, anchors(len_in))
                ramps.append(Ramp(round(intro[s][1]), round(intro[s][1] + len_in), rate_in, 1.0))
        if outro[s]:
            builder.hold(outro[s][0] - len_out, 1.0)  # the song's own tempo...
            if len_out:
                builder.ramp(outro[s][0], 1.0, rate_out, anchors(len_out))  # ...gliding to the next's
                ramps.append(Ramp(round(outro[s][0] - len_out), round(outro[s][0]), 1.0, rate_out))
            builder.hold(lengths[s], rate_out)
        else:
            builder.hold(lengths[s], 1.0)
        time_map = builder.build()

        out_start = round(float(time_map(round(intro[s][0])))) if intro[s] else 0
        out_end = int(time_map.dst[-1])
        if outro[s]:
            out_end = min(out_end, round(float(time_map(round(outro[s][0])))) + transitions[s].overlap)
        songs.append(RampSong(time_map, out_start, out_end, tuple(ramps)))
    return RampPlan(songs, transitions, fallback_crossfade)


def _fit_plain_crossfades(lengths: Sequence[int], overlaps: Sequence[int | None], fallback: int) -> list[int]:
    """Crossfade lengths: beat-matched overlaps as planned, plain ones fitted around them.

    overlaps[i] is the beat-matched overlap between tracks i and i+1, or None where
    they get a plain crossfade. A plain crossfade may use what its tracks have left
    after their other fade, or half the track if that's plain too (the Stage 1 half
    rule), so a track's fade-in and fade-out never overlap.
    """
    result = []
    for i, overlap in enumerate(overlaps):
        if overlap is not None:
            result.append(overlap)
            continue
        before = overlaps[i - 1] if i > 0 else None  # track i's fade-in, if beat-matched
        after = overlaps[i + 1] if i + 1 < len(overlaps) else None  # track i+1's fade-out
        room_a = lengths[i] - before if before is not None else lengths[i] // 2
        room_b = lengths[i + 1] - after if after is not None else lengths[i + 1] // 2
        result.append(max(0, min(fallback, room_a, room_b)))
    return result


def plan_crossfades(lengths: Sequence[int], crossfade: int) -> list[int]:
    """Return the overlap, in samples, between each pair of neighbouring tracks.

    Normally that's just `crossfade` every time, but a crossfade may use at most
    half of either track it touches. Why half? Track B in the diagram above has a
    fade-in AND a fade-out. If one crossfade could take more than half of B, the
    two fades would overlap inside B and three tracks would play at once. With the
    half rule, fade-in + fade-out <= length of B always holds: a short track's solo
    section can shrink to nothing, but never below zero. That's how a too-short
    song shrinks the crossfade instead of crashing the program.
    """
    if crossfade < 0:
        raise ValueError("crossfade can't be negative")
    return [min(crossfade, a // 2, b // 2) for a, b in zip(lengths, lengths[1:])]


def render_mix(
    tracks: Sequence[np.ndarray],
    crossfades: Sequence[int],
    transition: Transition | Sequence[Transition] = equal_power_crossfade,
) -> tuple[np.ndarray, list[int]]:
    """Lay `tracks` end to end, overlapping track i and i+1 by `crossfades[i]` samples.

    `transition` blends each overlap: one function for all of them, or a list with
    one per crossfade (Stage 4 picks a technique for each). Returns the mixed audio
    and the sample where each track starts. Nothing here limits the volume, so the
    result may contain peaks above 1.0.
    """
    if not tracks:
        raise ValueError("need at least one track to mix")
    if len(crossfades) != len(tracks) - 1:
        raise ValueError(f"{len(tracks)} tracks need {len(tracks) - 1} crossfades, got {len(crossfades)}")
    transitions = list(transition) if isinstance(transition, Sequence) else [transition] * len(crossfades)

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
            out[next_start : next_start + fade_out] = transitions[i](
                track[length - fade_out :], tracks[i + 1][:fade_out]
            )
    return out, starts


def mix_planned(
    tracks: Sequence[np.ndarray],
    plan: MixPlan,
    transition: Transition | Sequence[Transition] = equal_power_crossfade,
    ceiling_db: float = CEILING_DB,
    clicks: Sequence[np.ndarray] | None = None,
    sr: int = SAMPLE_RATE,
) -> Mix:
    """Render `plan` into one finished mix.

    `clicks`, if given, holds each track's beat positions (in samples, measured in
    that track) and puts an audible click on every one of them.
    """
    played = [track[start:end] for track, (start, end) in zip(tracks, plan.cuts)]
    audio, starts = render_mix(played, plan.crossfades, transition)
    if clicks is not None:
        _add_clicks(audio, starts, plan.cuts, clicks, sr)

    # Turn the whole mix down if any peak would clip; this is what
    # processing.prevent_clipping() does, but scaling in place (*=) skips making a
    # second full-length copy of the mix in memory. That's safe because render_mix
    # just created this buffer and nothing else uses it.
    gain = clipping_gain(audio, ceiling_db)
    if gain < 1.0:
        audio *= gain
    return Mix(audio, starts, list(plan.crossfades), safety_gain_db=gain_to_db(gain))


def mix_tracks(
    tracks: Sequence[np.ndarray],
    crossfade: int,
    transition: Transition = equal_power_crossfade,
    ceiling_db: float = CEILING_DB,
) -> Mix:
    """Stage 1: crossfade whole `tracks` together, in order, into one finished mix.

    `crossfade` is the requested overlap in samples (seconds * sample rate); it's
    shortened automatically wherever a track is too short for it.
    """
    return mix_planned(tracks, plan_fixed([len(track) for track in tracks], crossfade), transition, ceiling_db)


def _add_clicks(audio, starts, cuts, clicks, sr) -> None:
    """Add a click on every beat, in place (the buffer is the fresh one from render_mix).

    Odd and even songs click at different pitches. During a transition you hear both
    songs' clicks: if the beats are lined up, they hit together as one two-tone
    click; if not, you hear two separate clicks in quick succession (a "flam").
    """
    for i, (beats, start, (cut_start, cut_end)) in enumerate(zip(clicks, starts, cuts)):
        click = click_sound(sr, CLICK_FREQS[i % 2])
        for beat in beats:
            if cut_start <= beat < cut_end:  # only beats in the part of the track that plays
                position = start + round(beat) - cut_start  # track sample -> mix sample
                segment = audio[position : position + len(click)]
                segment += click[: len(segment), np.newaxis] if audio.ndim == 2 else click[: len(segment)]


# Mid-song transitions (Stage 4)
# ------------------------------

DROPSWAP_FADE_SECONDS = 0.004  # a drop swap's cut: this much crossfade, ending on the downbeat
DROPSWAP_SAFETY = 0.001  # ...and ending this long before the drop's first hit actually starts
ECHO_DELAY_BEATS = 1.0  # an echo repeats every beat (0.5 for a busier, half-beat echo)


@dataclass(frozen=True)
class CuedTransition:
    """One planned transition, in each song's own (unstretched) samples."""

    kind: str  # "blend", "dropswap", "echo" or "crossfade" (plain, end to start)
    exit: float  # A: where the overlap starts (blend), where it cuts (dropswap), where it stops (echo)
    entry: float  # B: where it starts playing (blend, echo), or the downbeat of its drop (dropswap)
    beats: int = 0  # blend: overlap length in beats; echo: tail length in A's beats
    tempo: TransitionTempo | None = None  # the meeting tempo (blend, dropswap)


@dataclass(frozen=True)
class CuedPlan:
    songs: list["RampSong"]  # each song's time map, and the part of it that plays
    transitions: list[CuedTransition]
    overlaps: list[int | None]  # samples each transition overlaps by (None: a plain crossfade, fitted later)
    swaps: list[int | None]  # blend: where the bass swaps, counted from the start of the overlap
    echo_delays: list[int | None]  # echo: one beat of the outgoing song, in samples
    exits: list[float] = field(default_factory=list)  # each transition's exit and entry after snapping
    entries: list[float] = field(default_factory=list)  # to the beat grid (song samples)
    fallback_crossfade: int = 0
    sr: int = SAMPLE_RATE

    def lengths(self) -> list[int]:
        """Each track's planned length, including an echo tail it ends with."""
        tails = [overlap if t.kind == "echo" else 0 for t, overlap in zip(self.transitions, self.overlaps)]
        return [song.length + tail for song, tail in zip(self.songs, [*tails, 0])]

    def mix_plan(self, lengths: Sequence[int] | None = None) -> MixPlan:
        lengths = list(lengths) if lengths is not None else self.lengths()
        return MixPlan(
            cuts=[(0, length) for length in lengths],
            crossfades=_fit_plain_crossfades(lengths, self.overlaps, self.fallback_crossfade),
            beat_aligned=[t.kind in ("blend", "dropswap") for t in self.transitions],
        )

    def transition_functions(self) -> list[Transition]:
        """The blend for each overlap (render_mix's `transition`)."""
        functions = []
        for t, swap in zip(self.transitions, self.swaps):
            if t.kind == "blend":
                functions.append(bass_swap(swap, self.sr))
            elif t.kind == "echo":
                functions.append(partial(echo_join, sr=self.sr))
            else:  # drop swap (a few ms) and plain crossfades
                functions.append(equal_power_crossfade)
        return functions


def plan_cued(
    lengths: Sequence[int],
    beats: Sequence[np.ndarray],
    bpms: Sequence[float | None],
    transitions: Sequence[CuedTransition],
    ramp_beats: int,
    fallback_crossfade: int,
    sr: int = SAMPLE_RATE,
    audio: Sequence[np.ndarray] | None = None,
) -> CuedPlan:
    """Plan a mix whose transitions start at the given cue points (see "Mid-song transitions").

    lengths, beats (in samples) and bpms describe each song as it is; transitions[i]
    joins song i to song i+1. Exits and entries are snapped onto each song's beat grid
    (like plan_beatmatched, the grid is fitted to the beats around the cue, so one
    misplaced beat can't throw it off).

    With `audio` (the songs themselves), a drop swap's cut is placed against the
    drops' actual first hits, which beat tracking puts a few ms late: see _onset_near.
    """
    count = len(lengths)
    periods = [60.0 * sr / bpm if bpm else None for bpm in bpms]
    intro: list[tuple | None] = [None] * count  # ("sync", entry, overlap end, rate) or ("free", entry)
    outro: list[tuple | None] = [None] * count  # ("sync", exit, rate, kind) or ("free", exit)
    overlaps, swaps, delays, exits, entries = [], [], [], [], []
    for a, t in enumerate(transitions):
        b = a + 1
        if t.kind in ("blend", "dropswap"):
            tempo = t.tempo
            period_b = periods[b] / tempo.factor  # B's beats, counted at the tempo it's matched as
            x = _snap_to_grid(beats[a], t.exit, periods[a])
            e = _snap_to_grid(octave_beats(beats[b], tempo.factor), t.entry, period_b)
            if t.kind == "dropswap" and audio is not None:
                # Cut just before each drop's real first hit, so B's drop lands where A's
                # would have, with its attack intact (and none of A's drop leaks through).
                safety = DROPSWAP_SAFETY * sr
                x = _onset_near(audio[a], x, sr) - safety
                e = _onset_near(audio[b], e, sr) - safety
            n = t.beats if t.kind == "blend" else 0
            # The overlap must fit inside A, give or take a bar (cues.END_SLACK_BEATS: past
            # the end there's only silence, which apply_time_map pads in).
            while x + (n - 4) * periods[a] > lengths[a] and x > periods[a]:
                x -= 4 * periods[a]
            outro[a] = ("sync", x, tempo.rate_out, t.kind)
            intro[b] = ("sync", e, e + n * period_b, tempo.rate_in)
            beat_out = 60.0 * sr / tempo.bpm  # samples per beat during the overlap
            overlaps.append(round(n * beat_out) if n else round(DROPSWAP_FADE_SECONDS * sr))
            swaps.append(round(n // 2 * beat_out) if n else None)  # beat n/2: a downbeat (n is whole bars)
            delays.append(None)
        elif t.kind == "echo":
            x, e = float(t.exit), float(t.entry)
            outro[a], intro[b] = ("free", x), ("free", e)
            overlaps.append(round(t.beats * periods[a]))
            swaps.append(None)
            delays.append(round(ECHO_DELAY_BEATS * periods[a]))
        else:  # a plain crossfade: A to its end, B from its start
            x, e = float(lengths[a]), 0.0
            overlaps.append(None)
            swaps.append(None)
            delays.append(None)
        exits.append(x)
        entries.append(e)

    songs = []
    for s in range(count):
        rate_in = intro[s][3] if intro[s] and intro[s][0] == "sync" else 1.0
        rate_out = outro[s][2] if outro[s] and outro[s][0] == "sync" else 1.0
        ramp_length = ramp_beats * periods[s] if periods[s] else 0.0
        len_in = ramp_length if rate_in != 1.0 else 0.0
        len_out = ramp_length if rate_out != 1.0 else 0.0
        room_start = (intro[s][2] if intro[s][0] == "sync" else intro[s][1]) if intro[s] else 0.0
        room_end = outro[s][1] if outro[s] else float(lengths[s])
        if len_in + len_out > room_end - room_start:  # not enough room: shrink both ramps to fit
            scale = max(0.0, room_end - room_start) / (len_in + len_out)
            len_in, len_out = len_in * scale, len_out * scale

        def anchors(length: float) -> int:
            return max(2, round(RAMP_ANCHORS_PER_BEAT * length / periods[s]))

        builder, ramps = TimeMapBuilder(), []
        if intro[s] and intro[s][0] == "sync":
            overlap_end = intro[s][2]
            builder.hold(overlap_end, rate_in)  # B's start (and the overlap) at the meeting tempo
            if len_in:
                builder.ramp(overlap_end + len_in, rate_in, 1.0, anchors(len_in))
                ramps.append(Ramp(round(overlap_end), round(overlap_end + len_in), rate_in, 1.0))
        if outro[s] and outro[s][0] == "sync":
            x = outro[s][1]
            builder.hold(x - len_out, 1.0)
            if len_out:
                builder.ramp(x, 1.0, rate_out, anchors(len_out))
                ramps.append(Ramp(round(x - len_out), round(x), 1.0, rate_out))
            builder.hold(lengths[s], rate_out)
        else:
            builder.hold(lengths[s], 1.0)
        time_map = builder.build()

        out_start = 0
        if intro[s] is not None:
            out_start = round(float(time_map(round(intro[s][1]))))
            if intro[s][0] == "sync" and transitions[s - 1].kind == "dropswap":
                out_start = max(0, out_start - overlaps[s - 1])  # the few ms before B's drop
        out_end = int(time_map.dst[-1])
        if outro[s] is not None:
            out_end = round(float(time_map(round(outro[s][1]))))
            if outro[s][0] == "sync" and outro[s][3] == "blend":
                out_end += overlaps[s]  # may run a little past the song's end: apply_time_map pads with silence
        songs.append(RampSong(time_map, out_start, out_end, tuple(ramps)))
    return CuedPlan(songs, list(transitions), overlaps, swaps, delays, exits, entries, fallback_crossfade, sr)


def _onset_near(audio: np.ndarray, position: float, sr: int, before: float = 0.025, after: float = 0.025) -> float:
    """Where the sound right at `position` actually starts: the first moment its level jumps.

    Beat tracking places beats on the peak of the onset curve, a few ms after a drum
    hit begins (on the test tracks, ~6 ms). For a blend that doesn't matter, but a
    drop swap cuts right at the beat, and cutting even 3 ms late would shave the
    attack off the incoming drop. So we look within `before`/`after` seconds of the
    beat for the first sample whose 1 ms loudness rises 20% of the way from the quiet
    just before to the peak of the hit. Returns `position` if there's no clear hit.
    """
    lo, hi = int(position - before * sr), int(position + after * sr)
    quiet_lo = int(position - 3 * before * sr)
    if quiet_lo < 0 or hi > len(audio):
        return float(position)
    mono = audio[quiet_lo:hi].mean(axis=1) if audio.ndim == 2 else audio[quiet_lo:hi]
    width = max(1, round(0.001 * sr))
    level = np.sqrt(np.convolve(mono.astype(np.float64) ** 2, np.ones(width) / width, mode="same"))
    search = level[lo - quiet_lo :]
    quiet, peak = float(np.median(level[: lo - quiet_lo])), float(search.max())
    if peak < 2 * quiet or peak <= 0:
        return float(position)
    first = int(np.argmax(search >= quiet + 0.2 * (peak - quiet)))
    return float(lo + first - width // 2)


def _snap_to_grid(beats: np.ndarray, position: float, period: float) -> float:
    """The beat-grid position nearest `position`, with the grid fitted to the beats around it."""
    near = beats[np.abs(beats - position) <= FIT_BEATS * period]
    if len(near) < 4:
        return float(position)
    offset, _ = _grid_offset(near, period)
    return float(offset + round((position - offset) / period) * period)
