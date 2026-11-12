"""Cue points and choosing them (cues.py), on made-up song structures: no audio needed."""

from dataclasses import replace

import numpy as np
import pytest

from analysis import MINOR_PROFILE
from cues import CueSettings, CueSong, all_options, blend_beats, choose_cues, find_cues
from ordering import TempoSettings
from structure import Section, Structure

RAMP = TempoSettings("ramp", "middle", 0.08)
A_MINOR = np.roll(MINOR_PROFILE, 9)  # a chroma vector that reads as A minor
EB_MINOR = np.roll(MINOR_PROFILE, 3)  # ...and one on the far side of the Camelot wheel


def song(name="s", bpm=128.0, layout=None, vocals=None, chroma=A_MINOR):
    """A made-up song: 8-bar phrases (15 s at 128 BPM), sections on phrase boundaries."""
    phrase = 32 * 60.0 / bpm
    layout = layout or ["intro", "build", "drop", "drop", "breakdown", "build", "drop", "drop", "outro"]
    numbers = {label: 0 for label in layout}
    counts = {label: layout.count(label) for label in layout}
    sections = []
    for k, label in enumerate(layout):
        numbers[label] += 1
        energy = {"drop": 1.0, "build": 0.5, "breakdown": 0.15, "intro": 0.1, "outro": 0.1}.get(label, 0.5)
        sections.append(Section(k * phrase, (k + 1) * phrase, label, numbers[label] if counts[label] > 1 else 0,
                                energy, energy, 4.0 if label == "build" else 0.0,
                                None if vocals is None else vocals.get(k, 0.0), chroma))
    end = len(layout) * phrase
    structure = Structure(np.arange(0, end, phrase / 8), np.arange(0, end, phrase), 4, sections)
    return CueSong(name, 0.0, end, bpm, structure, "A minor", 1.0)


def settings(min_play=60.0, kinds=("blend", "dropswap", "echo"), tempo=RAMP):
    return CueSettings(tempo, 8, 32, min_play, kinds)


def cued(*songs, s=None):
    s = s or settings()
    for x in songs:
        find_cues(x, s)
    return songs


def test_blends_last_whole_phrases():
    assert blend_beats(16, 8) == 32  # rounded up to one 8-bar phrase
    assert blend_beats(16, 4) == 16  # already a whole 4-bar phrase
    assert blend_beats(40, 8) == 64


def test_cue_points_are_the_musical_moments_and_all_on_phrase_boundaries():
    (a,) = cued(song())
    phrases = a.structure.phrases
    for cue in a.exits + a.entries:
        assert np.min(np.abs(phrases - cue.time)) < 1e-9
    # The end of every drop (even into another drop), each build's drop (for drop swaps), and the
    # end: here the end of drop 4 is also the last phrase where a blend still fits.
    assert [c.label for c in a.exits] == ["before drop 1", "end of drop 1", "end of drop 2", "before drop 3",
                                          "end of drop 3", "end of drop 4"]
    assert [c.label for c in a.entries] == ["intro", "build 1", "drop 1", "drop 2", "breakdown", "build 2",
                                            "drop 3", "drop 4"]


def test_min_play_rules_out_cues_too_close_to_either_end():
    a, b = cued(song("a"), song("b"))
    options = all_options(a, b, settings(min_play=60))
    assert options
    assert all(o.exit.time >= 60 for o in options)  # A has played 60 s before it's left
    assert all(b.end - o.entry.time >= 60 for o in options)  # B has 60 s left to play
    more = all_options(a, b, settings(min_play=100))
    assert all(o.exit.time >= 100 for o in more) and len(more) < len(options)


def test_a_drop_swap_goes_from_a_build_straight_into_a_drop():
    a, b = cued(song("a"), song("b"))
    swaps = [o for o in all_options(a, b, settings()) if o.kind == "dropswap"]
    assert swaps
    for o in swaps:
        assert o.exit.kind == "pre-drop" and o.exit.before.label == "build" and o.exit.after.label == "drop"
        assert o.entry.kind == "drop"


def test_the_best_blend_leaves_into_a_breakdown_or_outro_and_enters_at_a_build():
    a, b = cued(song("a"), song("b"))
    best = next(o for o in all_options(a, b, settings()) if o.kind == "blend")
    assert best.exit.after.label in ("breakdown", "outro")
    assert best.entry.after.label == "build"
    assert best.parts["best_parts"] == 0.0  # and it doesn't skip either song's drops


def test_a_blend_from_a_quiet_breakdown_into_a_full_drop_costs_more():
    a, b = cued(song("a"), song("b"))
    blends = [o for o in all_options(a, b, settings()) if o.kind == "blend"]
    quiet_to_drop = min(o.parts["energy"] for o in blends if o.exit.after.label == "outro" and o.entry.after.label == "drop")
    quiet_to_build = min(o.parts["energy"] for o in blends if o.exit.after.label == "outro" and o.entry.after.label == "build")
    assert quiet_to_drop > quiet_to_build + 0.5


def test_echo_is_the_fallback_when_the_tempos_are_too_far_apart():
    a, b = cued(song("a", bpm=128.0), song("b", bpm=150.0))  # ~8% each way, meeting in the middle
    options = all_options(a, b, settings())
    assert options and {o.kind for o in options} == {"echo"}
    assert options[0].parts.get("tempo", 0.0) == 0.0  # an echo needs no tempo change


def test_local_keys_at_the_cue_points_matter():
    same = all_options(*cued(song("a"), song("b")), settings())
    clash = all_options(*cued(song("a"), song("b", chroma=EB_MINOR)), settings())
    best_same = next(o for o in same if o.kind == "blend")
    best_clash = next(o for o in clash if o.kind == "blend")
    assert best_same.parts["harmony"] == 0.0 and best_clash.parts["harmony"] > 0.5


def test_two_vocals_at_once_is_a_big_penalty():
    vocal_everywhere = {k: 1.0 for k in range(9)}
    plain = next(o for o in all_options(*cued(song("a"), song("b")), settings()) if o.kind == "blend")
    singing = all_options(*cued(song("a", vocals=vocal_everywhere), song("b", vocals=vocal_everywhere)), settings())
    clashing = next(o for o in singing if o.kind == "blend")
    assert clashing.parts["vocals"] == pytest.approx(1.0)
    assert clashing.cost > plain.cost + 0.9


def test_transitions_option_restricts_the_type_and_falls_back_to_echo():
    a, b = cued(song("a"), song("b"))
    assert {o.kind for o in all_options(a, b, settings(kinds=("dropswap",)))} == {"dropswap"}
    no_build = song("c", layout=["intro", "drop", "drop", "breakdown", "drop", "drop", "outro"])
    find_cues(no_build, settings())
    assert {o.kind for o in all_options(no_build, b, settings(kinds=("dropswap",)))} == {"echo"}


def test_choosing_respects_min_play_for_songs_with_two_transitions():
    songs = cued(song("a"), song("b"), song("c"))
    s = settings(min_play=60)
    chain = [all_options(songs[0], songs[1], s), all_options(songs[1], songs[2], s)]
    chosen = choose_cues(list(songs), chain, s)
    played = chosen[1].exit_end - chosen[0].entry.time  # the middle song: entered by one, left by the other
    assert played >= 60
    assert chosen[0].exit_end - songs[0].start >= 60 and songs[2].end - chosen[1].entry.time >= 60


def test_auto_never_uses_the_same_type_three_times_in_a_row():
    songs = cued(*[song(name) for name in "abcdef"])
    s = settings(min_play=60)
    chain = [all_options(x, y, s) for x, y in zip(songs, songs[1:])]
    plain = choose_cues(list(songs), chain, s, variety=False)
    varied = choose_cues(list(songs), chain, s, variety=True)
    kinds = [o.kind for o in varied]
    assert all(len({kinds[i], kinds[i + 1], kinds[i + 2]}) > 1 for i in range(len(kinds) - 2))
    assert sum(o.cost for o in plain) <= sum(o.cost for o in varied)  # variety has a (small) price


def test_songs_without_a_beat_get_a_plain_crossfade_end_to_start():
    beatless = CueSong("ambient", 0.0, 120.0, None)
    a, b = cued(song("a"), beatless)
    options = all_options(a, b, settings())
    assert {o.kind for o in options} == {"crossfade"}
    assert b.entries[0].kind == "start"


def test_a_half_phrase_blend_is_offered_when_a_full_one_doesnt_fit():
    # At 90 BPM a 32-beat blend lasts 21 s, longer than this song's 4-bar outro.
    short_outro = song("a", bpm=90.0, layout=["intro", "build", "drop", "drop", "breakdown", "build", "drop", "drop"])
    tail = short_outro.structure
    last = tail.sections[-1]
    sections = tail.sections[:-1] + [replace(last, end=last.start + (last.end - last.start) / 2),
                                     replace(last, start=last.start + (last.end - last.start) / 2, label="outro",
                                             number=0, energy=0.1)]
    short_outro.structure = replace(tail, sections=sections)
    short_outro.end = sections[-1].end
    b = song("b", bpm=92.0)
    cued(short_outro, b)
    blends = [o for o in all_options(short_outro, b, settings()) if o.kind == "blend"]
    across_outro = [o for o in blends if o.exit.after is not None and o.exit.after.label == "outro"]
    assert across_outro and {o.beats for o in across_outro} == {16}  # only half a phrase fits there
    assert all(o.parts["base"] > 0 for o in across_outro)  # and it costs a little more than a full one


def test_entering_at_the_top_starts_on_the_first_phrase_not_a_pickup_beat():
    a = song("a")
    pickup = 2 * 60.0 / 128  # two beats before the first bar line
    a.structure.sections[0] = replace(a.structure.sections[0], start=-pickup)
    a.start = -pickup
    find_cues(a, settings())
    assert a.entries[0].time == a.structure.phrases[0] == 0.0
