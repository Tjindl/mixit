"""The optional neural-network tools: beat_this, demucs and allin1. Each test runs only if
its tool is installed (they need PyTorch, and download a model the first time)."""

import numpy as np
import pytest
import soundfile as sf

import automix
import structure
from analysis import AnalysisCache, beat_this_available, beat_this_beats
from audio_io import SAMPLE_RATE as SR

from conftest import make_structured_song

pytestmark = pytest.mark.neural


@pytest.mark.skipif(not beat_this_available(), reason="beat_this isn't installed")
def test_beat_this_finds_the_bars():
    audio, info = make_structured_song(126.0, (("pad", 8), ("drop", 8), ("pad", 8), ("drop", 8)), lead_in_beats=2)
    found = beat_this_beats(audio)
    assert found is not None
    beats, downbeats = found
    bar_lines = np.append(info["downbeats"], info["boundaries"][-1])
    loud = (downbeats.times > info["boundaries"][1]) & (downbeats.times < info["boundaries"][2])  # in drop 1
    assert loud.sum() >= 6
    assert max(np.min(np.abs(bar_lines - t)) for t in downbeats.times[loud]) < 0.03
    assert downbeats.method == "beat_this" and downbeats.beats_per_bar == 4


@pytest.mark.skipif(not structure.demucs_available(), reason="demucs isn't installed")
def test_demucs_separates_vocals_once_and_scores_sections(tmp_path):
    audio, _ = make_structured_song(126.0, (("pad", 8), ("drop", 8), ("pad", 8), ("drop", 8)))
    path = tmp_path / "instrumental.wav"
    sf.write(path, audio, SR)
    first, _ = automix.analyze(path, AnalysisCache(tmp_path / "cache.json"), audio)
    stems = list(structure.stems_dir().glob("instrumental-*.vocals.flac"))
    assert len(stems) == 1  # the stem is kept...
    second = structure.vocal_levels(path, first.beats)  # ...and reused, not separated again
    np.testing.assert_allclose(second, first.structure.vocal_db, atol=0.01)
    sections = structure.sections(first.structure, 8).sections
    assert all(s.vocals is not None and s.vocals < 0.5 for s in sections)  # no singing in this one


@pytest.mark.skipif(not structure.allin1_available(), reason="allin1 isn't installed")
def test_allin1_sections_are_used_and_snapped_to_phrases(tmp_path):
    audio, info = make_structured_song(126.0, (("pad", 8), ("drop", 8), ("pad", 8), ("drop", 8)))
    path = tmp_path / "song.wav"
    sf.write(path, audio, SR)
    analysis, _ = automix.analyze(path, AnalysisCache(tmp_path / "cache.json"), audio)
    assert analysis.structure.segment_method == "allin1"
    result = structure.sections(analysis.structure, 8)
    for section in result.sections[1:]:
        assert np.min(np.abs(result.phrases - section.start)) < 1e-6
