"""Downbeats, phrases and sections (Stage 4), on synthetic songs built from known sections."""

import numpy as np
import pytest

from analysis import AnalysisCache, SongAnalysis, detect_beats, find_downbeats
from structure import (
    StructureFeatures,
    analyze_structure,
    checkerboard_kernel,
    novelty_curve,
    phrase_starts,
    sections,
)

from conftest import make_structured_song

FOUR_SECTIONS = (("pad", 8), ("drop", 8), ("pad", 8), ("drop", 8))
FULL_SONG = (("pad", 8), ("build", 8), ("drop", 16), ("pad", 8), ("build", 8), ("drop", 16), ("pad", 8))


def analyzed(song_sections, lead_in_beats=1, bpm=126.0, seed=1):
    audio, info = make_structured_song(bpm, song_sections, lead_in_beats=lead_in_beats, seed=seed)
    found = detect_beats(audio)
    downbeats = find_downbeats(audio, found.beats)
    return audio, info, found, downbeats, analyze_structure(audio, found.beats, downbeats)


@pytest.fixture(scope="module")
def four_sections():
    return analyzed(FOUR_SECTIONS)


@pytest.fixture(scope="module")
def full_song():
    return analyzed(FULL_SONG, lead_in_beats=3)


@pytest.mark.parametrize("lead_in_beats", [1, 2, 3])
def test_the_downbeat_fallback_finds_the_bar_starts(lead_in_beats):
    # The song starts 1-3 beats before its first bar line, so the first beat isn't
    # beat 1: only the accented kick on beat 1 says where the bars start.
    audio, info, found, downbeats, _ = analyzed(FOUR_SECTIONS, lead_in_beats=lead_in_beats)
    bar_lines = np.append(info["downbeats"], info["boundaries"][-1])  # plus the bar line after the last bar
    errors = [np.min(np.abs(bar_lines - t)) for t in downbeats.times]
    assert max(errors) < 0.02  # every downbeat found is on a bar line (within 20 ms)...
    assert len(downbeats.times) >= len(info["downbeats"]) - 1  # ...and (almost) every bar line is found
    assert downbeats.method == "bass" and downbeats.beats_per_bar == 4


@pytest.mark.parametrize("song", ["four_sections", "full_song"])
def test_segmentation_finds_the_section_boundaries_within_one_bar(song, request):
    audio, info, found, downbeats, features = request.getfixturevalue(song)
    structure = sections(features, phrase_bars=8, duration=len(audio) / 44_100)
    found_edges = [s.start for s in structure.sections[1:]]
    true_edges = info["boundaries"][1:-1]
    bar = 4 * info["beat"]
    assert len(found_edges) == len(true_edges)
    assert max(np.min(np.abs(np.array(found_edges) - t)) for t in true_edges) < bar


def test_sections_are_labelled_by_energy_bass_and_position(four_sections, full_song):
    for (audio, info, *_, features), expected in [
        (four_sections, ["intro", "drop 1", "breakdown", "drop 2"]),
        (full_song, ["intro", "build 1", "drop 1", "breakdown", "build 2", "drop 2", "outro"]),
    ]:
        structure = sections(features, phrase_bars=8, duration=len(audio) / 44_100)
        assert [s.name for s in structure.sections] == expected
    drops = [s for s in structure.sections if s.label == "drop"]
    builds = [s for s in structure.sections if s.label == "build"]
    assert all(d.energy > 0.8 for d in drops) and all(b.rise_db > 2 for b in builds)


def test_phrases_are_every_8_bars_starting_where_the_sections_do(full_song):
    audio, info, found, downbeats, features = full_song
    phrases = phrase_starts(features, phrase_bars=8)
    true_phrases = info["downbeats"][::8]  # every section here is a whole number of 8-bar phrases
    assert len(phrases) == len(true_phrases)
    assert np.max(np.abs(phrases - true_phrases)) < 0.02
    four_bar = phrase_starts(features, phrase_bars=4)
    assert len(four_bar) == pytest.approx(2 * len(phrases), abs=1)


def test_every_section_starts_on_a_phrase_boundary(full_song):
    audio, info, found, downbeats, features = full_song
    structure = sections(features, phrase_bars=8, duration=len(audio) / 44_100)
    for section in structure.sections[1:]:
        assert np.min(np.abs(structure.phrases - section.start)) < 1e-6


def test_the_checkerboard_kernel_finds_the_corner_between_two_blocks():
    kernel = checkerboard_kernel(4)
    assert kernel.shape == (8, 8)
    assert kernel[0, 0] > 0 and kernel[-1, -1] > 0 and kernel[0, -1] < 0  # +1 past/past, -1 past/future
    # A similarity matrix with two "sections": beats 0-19 alike, beats 20-49 alike.
    similarity = np.zeros((50, 50))
    similarity[:20, :20] = similarity[20:, 20:] = 1.0
    novelty = novelty_curve(similarity, half=8)
    assert int(np.argmax(novelty)) == 20
    assert novelty[5] < 0.1 and novelty[35] < 0.1  # flat inside a section


def test_structure_and_downbeats_survive_the_cache(tmp_path, four_sections):
    audio, info, found, downbeats, features = four_sections
    song = tmp_path / "song.wav"
    song.write_bytes(b"pretend audio")
    saved = SongAnalysis(found.bpm, found.beats, key="A minor", key_confidence=0.9, key_method="librosa",
                         downbeats=downbeats, structure=features)
    AnalysisCache(tmp_path / "cache.json").put(song, saved)
    loaded = AnalysisCache(tmp_path / "cache.json").get(song)
    np.testing.assert_allclose(loaded.downbeats.times, downbeats.times, atol=1e-5)
    assert isinstance(loaded.structure, StructureFeatures)
    np.testing.assert_allclose(loaded.structure.novelty, features.novelty, atol=1e-3)
    assert loaded.structure.chroma.shape == features.chroma.shape
    original = [s.name for s in sections(features, 8, len(audio) / 44_100).sections]
    assert [s.name for s in sections(loaded.structure, 8, len(audio) / 44_100).sections] == original


def test_vocal_presence_comes_from_the_vocal_stem(four_sections):
    from dataclasses import replace

    audio, info, found, downbeats, features = four_sections
    vocal_db = features.energy_db - 30.0  # bleed only...
    in_drop_2 = features.beat_times >= info["boundaries"][3]
    vocal_db[in_drop_2] = features.energy_db[in_drop_2] - 4.0  # ...except a lead vocal in the second drop
    structure = sections(replace(features, vocal_db=vocal_db), 8, len(audio) / 44_100)
    assert [round(s.vocals) for s in structure.sections] == [0, 0, 0, 1]
    assert all(s.vocals is None for s in sections(features, 8, len(audio) / 44_100).sections)  # no demucs
