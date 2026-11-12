"""Cue points: where each song is left and entered, and how (Stage 4).

A DJ doesn't wait for a song to end. They leave it at a good moment (after a drop,
going into a breakdown) and bring the next one in at a good moment (at the start of
a build, so its drop hits right after the transition), using whichever technique
suits the two moments. This module does the same in three steps:

    1. candidates   exit_cues() / entry_cues(): the musically sensible points in
                    each song, all on phrase boundaries (structure.py)
    2. scoring      score_option(): for one exit in A, one entry in B and one
                    transition type, how good would it be? (lower = better)
    3. choosing     best_options() gives each ordered pair of songs its best option,
                    whose cost the ordering (ordering.py) uses; choose_cues() then
                    picks a consistent set for the final order: each song entered
                    before it's left, played for at least --min-play seconds, and
                    no transition type three times in a row

All the costs and weights are in the block just below.
"""

import math
from dataclasses import dataclass, field

from analysis import camelot
from ordering import KEY_COSTS, TEMPO_FULL_COST_PERCENT, UNMIXABLE_COST, TempoSettings, key_relation, label_for
from structure import Section, Structure
from tempo import TransitionTempo, transition_tempo

# Costs and weights
# =================

TRANSITION_KINDS = ("blend", "dropswap", "echo")  # what --transitions can ask for ("crossfade" is the fallback)


@dataclass(frozen=True)
class CueWeights:
    harmony: float = 1.0  # local keys at the cue points
    tempo: float = 1.0  # tempo change needed
    energy: float = 1.0  # energy flow across the transition
    vocals: float = 1.0  # two vocals at once
    best_parts: float = 0.8  # drops/choruses a song doesn't get to play


CUE_WEIGHTS = CueWeights()

# Blends are what make a mix sound mixed: both songs playing together, beats locked,
# before one gives way. A drop swap is a cut (exciting, but no overlap), and an echo
# out always works (nothing has to line up) but sounds the most like one song simply
# stopping and the next starting. So echo starts at 0.9: it wins only where blends
# and drop swaps are risky (a big key clash, or tempos too far apart to beat-match).
BASE_COST = {"blend": 0.0, "dropswap": 0.25, "echo": 0.9, "crossfade": 1.0}
HARMONY_SHARE = {"blend": 1.0, "dropswap": 0.5, "echo": 0.5, "crossfade": 0.5}  # how much key clashes matter

# Blend: what A plays while it fades out (the section after its exit), and what B
# starts with. Fading out a breakdown or outro is ideal; fading out a drop wastes it.
# Entering at a build is ideal: B's drop hits right after the transition.
BLEND_OUT_COST = {"breakdown": 0.0, "outro": 0.0, "groove": 0.2, "intro": 0.2, "build": 0.4, "drop": 0.6}
BLEND_IN_COST = {"build": 0.0, "intro": 0.05, "start": 0.05, "breakdown": 0.15, "groove": 0.3, "drop": 0.5}
BLEND_ENERGY_SCALE = 1.0  # an energy jump this big (0..1 scale) across a blend costs 1.0 (squared)
NO_DROP_AFTER_BLEND = 0.2  # a blend whose end isn't followed by B's drop (the ideal: B's drop hits right after)
# Echo out: what A was playing when it echoes away, and what B comes in with.
ECHO_OUT_COST = {"drop": 0.0, "breakdown": 0.1, "groove": 0.1, "build": 0.1, "outro": 0.0, "intro": 0.3}
ECHO_IN_COST = {"drop": 0.0, "build": 0.05, "intro": 0.1, "start": 0.1, "breakdown": 0.15, "groove": 0.2}
VOCAL_CLASH_COST = 1.0  # both vocals fully up front during a blend (an echo's tail counts half)

REPEAT_PENALTY = 1.0  # --transitions auto: added when a type would be used 3 times in a row
BLEND_PHRASES = 1  # a blend lasts this many phrases (--crossfade-beats, rounded up to whole phrases)
SHORT_BLEND_COST = 0.1  # a half-phrase blend (16 beats), for songs without room for a full one
ECHO_BEATS = 8  # the echo tail lasts this many of the outgoing song's beats
END_SLACK_BEATS = 4  # a blend may run this far past the end of a song's audio (silence: it's over)
DROPSWAP_FADE = 0.004  # seconds of crossfade at a drop swap's cut


# Cue points
# ==========


@dataclass(frozen=True)
class Cue:
    """A point in a song (seconds, the song's own timeline) where a transition can start."""

    time: float
    kind: str  # exits: "drop-end", "breakdown", "outro", "pre-drop", "end"; entries: section labels, "start"
    label: str  # for people: "end of drop 2", "build", "before drop 1", ...
    after: Section | None = None  # the section that starts here
    before: Section | None = None  # the section that ends here


@dataclass
class CueSong:
    """What cue planning needs to know about one song."""

    name: str
    start: float  # its first beat (seconds): where it starts if it starts at the top
    end: float  # its last beat, or the end of the audio for a song with no beat
    bpm: float | None
    structure: Structure | None = None  # None: no usable structure (start/end cues only)
    key: str | None = None  # the whole song's key, for songs without structure
    key_confidence: float = 0.0
    exits: list[Cue] = field(default_factory=list)
    entries: list[Cue] = field(default_factory=list)

    @property
    def beat(self) -> float:
        return 60.0 / self.bpm if self.bpm else 0.0

    def drops(self) -> list[Section]:
        return [s for s in self.structure.sections if s.label == "drop"] if self.structure else []


@dataclass(frozen=True)
class CueSettings:
    tempo: TempoSettings
    phrase_bars: int = 8
    blend_beats: int = 32  # the blend length, a whole number of phrases
    min_play: float = 60.0  # seconds
    kinds: tuple[str, ...] = TRANSITION_KINDS  # allowed transition types (--transitions)
    weights: CueWeights = CUE_WEIGHTS


def blend_beats(crossfade_beats: int, phrase_bars: int, beats_per_bar: int = 4) -> int:
    """--crossfade-beats rounded up to whole phrases, so a blend starts AND ends on a phrase boundary."""
    phrase = phrase_bars * beats_per_bar
    return phrase * max(BLEND_PHRASES, math.ceil(crossfade_beats / phrase))


def find_cues(song: CueSong, settings: CueSettings) -> None:
    """Fill in song.exits and song.entries (see the module docstring for what makes a cue)."""
    song.exits, song.entries = [], []
    blend_length = settings.blend_beats * song.beat
    sections = song.structure.sections if song.structure else []
    long_enough = song.end - song.start >= settings.min_play + blend_length
    if sections and long_enough:
        for before, after in zip(sections, sections[1:]):
            t = after.start
            if before.label == "drop":
                song.exits.append(Cue(t, "drop-end", f"end of {before.name}", after, before))
            elif after.label == "breakdown":
                song.exits.append(Cue(t, "breakdown", f"start of {after.name}", after, before))
            elif after.label == "outro":
                song.exits.append(Cue(t, "outro", "start of outro", after, before))
            if after.label == "drop" and before.label == "build":
                song.exits.append(Cue(t, "pre-drop", f"before {after.name}", after, before))
        # Enter at the top on the first phrase boundary, not the first beat: a song can
        # open with a beat or two of pickup before its first bar, and starting a blend
        # there would put its bars out of step with the other song's.
        first = song.structure.phrases[0] if len(song.structure.phrases) else sections[0].start
        first = first if first < sections[0].end else sections[0].start
        song.entries.append(Cue(float(first), "intro", sections[0].name if sections[0].label == "intro"
                                else f"start, in {sections[0].name}", sections[0], None))
        for before, after in zip(sections, sections[1:]):
            if after.label in ("build", "breakdown", "drop"):
                song.entries.append(Cue(after.start, after.label, after.name, after, before))
    # The song's very end, as in Stage 3: the last phrase where a blend still fits,
    # or (without structure) the last beat, minus the blend.
    if song.structure is not None and len(song.structure.phrases):
        fits = [p for p in song.structure.phrases if p + blend_length - END_SLACK_BEATS * song.beat <= song.end]
        end_time = float(fits[-1]) if fits else song.end - blend_length
    else:
        end_time = song.end - blend_length if song.bpm else song.end
    if not any(abs(c.time - end_time) < 0.01 for c in song.exits):
        section = song.structure.section_at(end_time) if song.structure else None
        label = "end" if section is None or section.label == "outro" else f"end, in {section.name}"
        song.exits.append(Cue(end_time, "end", label, section, section))
    if not song.entries:
        section = sections[0] if sections else None
        song.entries.append(Cue(song.start, "start", "start", section, None))
    song.exits.sort(key=lambda c: c.time)
    song.entries.sort(key=lambda c: c.time)


# Scoring
# =======


@dataclass(frozen=True)
class CueOption:
    """One way to go from song A to song B: where to leave A, where to enter B, and how."""

    exit: Cue
    entry: Cue
    kind: str  # "blend", "dropswap", "echo", or "crossfade" (plain, for songs with no beat)
    cost: float
    parts: dict = field(default_factory=dict)
    tempo: TransitionTempo | None = None  # where the two songs meet (blend, dropswap)
    beats: int = 0  # blend: overlap length; echo: tail length (A's beats)

    @property
    def label(self) -> str:
        return label_for(self.cost)

    @property
    def exit_end(self) -> float:
        """When A stops playing (seconds, A's timeline), before any tempo change."""
        return self.exit.time + (self.beats * (60.0 / self.tempo.bpm * self.tempo.rate_out)
                                 if self.kind == "blend" and self.tempo else 0.0)


def _section_key_cost(a: Section | None, b: Section | None, song_a: CueSong, song_b: CueSong) -> float:
    """The Camelot cost of the two sections' local keys, scaled by how sure we are of them."""
    if a is not None and b is not None:
        key_a, key_b = a.key, b.key
        names, confidence = (key_a.key, key_b.key), min(key_a.confidence, key_b.confidence)
    else:  # no structure: fall back on the songs' overall keys
        names, confidence = (song_a.key, song_b.key), min(song_a.key_confidence, song_b.key_confidence)
    if None in names:
        return 0.0
    return KEY_COSTS[key_relation(camelot(names[0]), camelot(names[1]))] * confidence


def _missed(song: CueSong, after: float | None = None, before: float | None = None,
            spare: Section | None = None) -> float:
    """The share of the song's drop time that's skipped: after time `after`, or before `before`."""
    drops = [d for d in song.drops() if d is not spare]
    total = sum(d.end - d.start for d in song.drops())
    if not total:
        return 0.0
    if after is not None:
        missed = sum(max(0.0, d.end - max(d.start, after)) for d in drops)
    else:
        missed = sum(max(0.0, min(d.end, before) - d.start) for d in drops)
    return missed / total


def _tempo(a: CueSong, b: CueSong, settings: CueSettings) -> TransitionTempo | None:
    """The meeting tempo for a beat-synced transition, or None if the songs can't be beat-matched."""
    if not (a.bpm and b.bpm) or settings.tempo.mode == "none":
        return None
    if settings.tempo.mode == "global":  # both songs already stretched to the mix tempo (or not at all)
        return TransitionTempo(a.bpm, 1.0, 1.0) if abs(a.bpm / b.bpm - 1) < 1e-6 else None
    tempo = transition_tempo(a.bpm, b.bpm, settings.tempo.meet)
    return tempo if tempo.stretch <= settings.tempo.max_stretch else None


def blend_lengths(settings: CueSettings) -> tuple[int, ...]:
    """The blend lengths to try, in beats: a full phrase, and half of one if that's still 16+ beats.

    A full 8-bar phrase is 32 beats: 15 s at 128 BPM, but 21 s at 90 BPM, longer
    than many pop songs' intros and outros. The half-phrase blend still starts on a
    phrase boundary, and fits them.
    """
    half = settings.blend_beats // 2
    return (settings.blend_beats, half) if half >= 16 else (settings.blend_beats,)


def score_option(a: CueSong, b: CueSong, exit: Cue, entry: Cue, kind: str, settings: CueSettings,
                 stretch: float = 0.0, beats: int | None = None) -> CueOption | None:
    """How good is leaving A at `exit` and entering B at `entry` with a `kind` transition? None if impossible.

    `stretch` is the tempo change the pair needs anyway (global mode: each song's
    stretch to the mix tempo); ramp mode works it out here. `beats` is a blend's
    length (default: a full phrase).
    """
    w = settings.weights
    tempo = _tempo(a, b, settings) if kind in ("blend", "dropswap") else None
    if kind in ("blend", "dropswap") and tempo is None:
        return None
    if kind == "dropswap" and not (exit.kind == "pre-drop" and entry.kind == "drop"):
        return None
    if kind == "blend" and exit.kind == "pre-drop":
        return None
    if kind == "echo" and not a.bpm:
        return None
    if kind == "crossfade" and not (exit.kind == "end" and entry.kind in ("start", "intro")):
        return None
    if kind == "blend":
        beats = beats or settings.blend_beats
    else:
        beats = ECHO_BEATS if kind == "echo" else 0

    # Room in each song: A needs the blend's length after its exit, B after its entry.
    # (An outro is often exactly a phrase long and ends on its last beat, so a blend
    # across it ends right at the end of the audio: allow up to a bar of slack.)
    if kind == "blend":
        if exit.time + (beats - END_SLACK_BEATS) * a.beat > a.end or entry.time + (beats - END_SLACK_BEATS) * b.beat > b.end:
            return None
    # --min-play, as far as one transition can tell: A from its very start, B to its very end.
    a_short = a.end - a.start < settings.min_play + settings.blend_beats * a.beat
    b_short = b.end - b.start < settings.min_play + settings.blend_beats * b.beat
    if not a_short and exit.time - a.start < settings.min_play:
        return None
    if not b_short and b.end - entry.time < settings.min_play:
        return None

    parts = {"base": BASE_COST[kind] + (SHORT_BLEND_COST if kind == "blend" and beats < settings.blend_beats else 0.0)}
    # Harmony: the local keys of what's heard together (blend, echo) or back to back (drop swap).
    heard_a = exit.after if kind == "blend" else exit.before
    parts["harmony"] = w.harmony * HARMONY_SHARE[kind] * _section_key_cost(heard_a, entry.after, a, b)
    if tempo is not None:
        parts["tempo"] = w.tempo * tempo.stretch * 100 / TEMPO_FULL_COST_PERCENT
    elif stretch:
        parts["tempo"] = w.tempo * stretch * 100 / TEMPO_FULL_COST_PERCENT

    # Energy flow.
    out_label = (exit.after.label if exit.after else "outro") if kind == "blend" else (exit.before.label if exit.before else "drop")
    in_label = entry.after.label if entry.after else "start"
    if kind == "blend":
        e_a = exit.after.energy if exit.after else 0.3
        e_b = entry.after.energy if entry.after else 0.3
        after = b.structure.section_at(entry.time + beats * b.beat + 1e-3) if b.structure else None
        parts["energy"] = w.energy * (BLEND_OUT_COST.get(out_label, 0.2) + BLEND_IN_COST.get(in_label, 0.2)
                                      + ((e_b - e_a) / BLEND_ENERGY_SCALE) ** 2
                                      + (0.0 if after is not None and after.label == "drop" else NO_DROP_AFTER_BLEND))
    elif kind == "dropswap":  # A's drop and B's drop should be similar in intensity
        parts["energy"] = w.energy * ((entry.after.energy - exit.after.energy) / BLEND_ENERGY_SCALE) ** 2
    elif kind == "echo":
        parts["energy"] = w.energy * (ECHO_OUT_COST.get(out_label, 0.2) + ECHO_IN_COST.get(in_label, 0.2))
    else:
        parts["energy"] = 0.0

    # Vocals: two singers at once is the worst clash.
    vocal_a = (exit.after if kind == "blend" else exit.before)
    if kind in ("blend", "echo") and vocal_a is not None and entry.after is not None \
            and vocal_a.vocals is not None and entry.after.vocals is not None:
        share = 1.0 if kind == "blend" else 0.5
        parts["vocals"] = w.vocals * VOCAL_CLASH_COST * share * min(vocal_a.vocals, entry.after.vocals)

    # The best parts: drops that A or B won't get to play. (In a drop swap, B's drop
    # replaces A's, so A's swapped drop doesn't count as missed.)
    spare = exit.after if kind == "dropswap" else None
    parts["best_parts"] = w.best_parts * (_missed(a, after=exit.time, spare=spare) + _missed(b, before=entry.time))
    return CueOption(exit, entry, kind, sum(parts.values()), parts, tempo, beats)


def all_options(a: CueSong, b: CueSong, settings: CueSettings, stretch: float = 0.0) -> list[CueOption]:
    """Every possible way from A to B, cheapest first (restricted to --transitions if possible)."""
    if not (a.bpm and b.bpm):  # no beat: only a plain crossfade, end to start
        kinds = ("crossfade",)
    else:
        kinds = TRANSITION_KINDS
    options = [o for x in a.exits for e in b.entries for k in kinds
               for n in (blend_lengths(settings) if k == "blend" else (None,))
               if (o := score_option(a, b, x, e, k, settings, stretch, n)) is not None]
    wanted = [o for o in options if o.kind in settings.kinds]
    if not wanted:  # the asked-for type is impossible here: the safe echo, then a plain crossfade
        wanted = [o for o in options if o.kind == "echo"] or [o for o in options if o.kind == "crossfade"]
    if not wanted:
        end = next(x for x in reversed(a.exits))
        start = b.entries[0]
        wanted = [CueOption(end, start, "crossfade", UNMIXABLE_COST, {"base": UNMIXABLE_COST})]
    return sorted(wanted, key=lambda o: o.cost)


# Choosing
# ========
# best_options() is what the ordering sees: for each ordered pair (A, B), the cost of
# its best option. That treats each transition on its own, but one song's two
# transitions are tied together: its exit has to come after its entry, with at
# least --min-play seconds between them. So once the order is fixed, choose_cues()
# picks the whole chain at once with dynamic programming (as in beat tracking,
# analysis.py step 3): going through the transitions in order, it keeps, for every
# option of transition i (and how many times in a row its type has been used), the
# cheapest consistent way to reach it. Usually that's just each pair's best option.


def best_options(songs: list[CueSong], settings: CueSettings, stretches=None) -> dict:
    """{(i, j): cheapest options from song i into song j}, for every ordered pair."""
    result = {}
    for i, a in enumerate(songs):
        for j, b in enumerate(songs):
            if i != j:
                result[i, j] = all_options(a, b, settings, stretches[i][j] if stretches else 0.0)
    return result


def _plays_enough(song: CueSong, entry: CueOption | None, exit: CueOption | None, settings: CueSettings) -> bool:
    """Does `song` play for at least --min-play between its entry and its exit (or as long as it can)?"""
    start = entry.entry.time if entry else song.start
    stop = exit.exit_end if exit else song.end
    if stop <= start + (entry.beats * song.beat if entry and entry.kind == "blend" else 0.0):
        return False
    whole = song.end - song.start
    return stop - start >= min(settings.min_play, whole - settings.blend_beats * 2 * song.beat)


def choose_cues(songs: list[CueSong], options: list[list[CueOption]], settings: CueSettings,
                variety: bool = True) -> list[CueOption]:
    """The cheapest consistent option for each transition along the given order (see "Choosing")."""
    if not options:
        return []
    # best[(k, run)] = (cost, path): the cheapest way through transition i ending in
    # option k, whose type has now been used `run` times in a row.
    best = {}
    for k, option in enumerate(options[0]):
        if _plays_enough(songs[0], None, option, settings):
            best[k, 1] = (option.cost, [k])
    for i in range(1, len(options)):
        nxt = {}
        for (k, run), (cost, path) in best.items():
            previous = options[i - 1][k]
            for m, option in enumerate(options[i]):
                if not _plays_enough(songs[i], previous, option, settings):
                    continue
                streak = run + 1 if option.kind == previous.kind else 1
                total = cost + option.cost + (REPEAT_PENALTY if variety and streak >= 3 else 0.0)
                key = (m, min(streak, 3))
                if key not in nxt or total < nxt[key][0]:
                    nxt[key] = (total, path + [m])
        best = nxt
    finals = {key: value for key, value in best.items()
              if _plays_enough(songs[-1], options[-1][key[0]], None, settings)}
    if not finals:  # nothing satisfies --min-play everywhere: take the cheapest chain regardless
        if not best:
            return [o[0] for o in options]
        finals = best
    _, path = min(finals.values(), key=lambda v: v[0])
    return [options[i][k] for i, k in enumerate(path)]
