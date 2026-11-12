"""Key detection, Camelot notation, energy scores, and caching them (Stage 3).

The key tests run with librosa's method, and again with essentia's if it's installed.
"""

import importlib.util
import json

import numpy as np
import pytest

import analysis
from analysis import (
    AnalysisCache,
    EnergyFeatures,
    SongAnalysis,
    camelot,
    energy_features,
    energy_scores,
    estimate_key,
    key_from_chroma,
    parse_key,
)
from audio_io import SAMPLE_RATE as SR

C, D, E, F, G, A, B = 60, 62, 64, 65, 67, 69, 71  # MIDI note numbers around middle C


def note(midi, seconds, amplitude=0.2):
    """A plucked note: a sine with two overtones (like a real instrument), fading out."""
    t = np.arange(round(seconds * SR)) / SR
    freq = 440.0 * 2 ** ((midi - 69) / 12)
    envelope = np.minimum(1.0, t / 0.01) * np.exp(-t / (0.8 * seconds))
    return amplitude * envelope * sum(np.sin(2 * np.pi * freq * h * t) / h for h in (1, 2, 3))


def play(groups, seconds, transpose=0):
    """Play each group of notes (a chord, or one note) in turn; returns stereo float32."""
    mono = np.concatenate([sum(note(m + transpose, seconds) for m in group) for group in groups])
    return np.column_stack([mono, mono]).astype(np.float32)


MAJOR_CHORDS = [[C, E, G], [F, A, C + 12], [G, B, D + 12], [C, E, G]] * 4  # I IV V I
MAJOR_SCALE = [[n] for n in (C, D, E, F, G, A, B, C + 12)] * 3
MINOR_CHORDS = [[A - 12, C, E], [D, F, A], [E, G + 1, B], [A - 12, C, E]] * 4  # i iv V i in A minor
MINOR_SCALE = [[n] for n in (A - 12, B - 12, C, D, E, F, G + 1, A)] * 3  # A harmonic minor


@pytest.fixture(params=["librosa", "essentia"])
def key_detector(request, monkeypatch):
    """Run a test with each key detector (essentia's only if it's installed)."""
    if request.param == "essentia" and importlib.util.find_spec("essentia") is None:
        pytest.skip("essentia isn't installed")
    monkeypatch.setattr(analysis, "key_method", lambda: request.param)
    return request.param


@pytest.mark.parametrize("groups, seconds, expected", [
    (MAJOR_CHORDS, 1.0, "C major"),
    (MAJOR_SCALE, 0.4, "C major"),
    (MINOR_CHORDS, 1.0, "A minor"),
    (MINOR_SCALE, 0.4, "A minor"),
])
def test_finds_the_key_of_chords_and_scales(key_detector, groups, seconds, expected):
    found = estimate_key(play(groups, seconds))
    assert (found.key, found.method) == (expected, key_detector)
    assert found.confidence > 0.8  # these are as clear as keys get


@pytest.mark.parametrize("semitones, major, minor", [(7, "G major", "E minor"), (6, "F# major", "Eb minor"),
                                                     (-2, "Bb major", "G minor")])
def test_finds_transposed_keys_too(key_detector, semitones, major, minor):
    # Every key profile is a rotation of the same pattern, so any root should work.
    assert estimate_key(play(MAJOR_CHORDS, 1.0, semitones)).key == major
    assert estimate_key(play(MINOR_CHORDS, 1.0, semitones)).key == minor


@pytest.mark.parametrize("seed", range(4))
def test_noise_and_silence_have_no_confident_key(key_detector, seed):
    noise = (0.1 * np.random.default_rng(seed).standard_normal((10 * SR, 2))).astype(np.float32)
    assert estimate_key(noise).confidence < 0.2
    silence = estimate_key(np.zeros((SR, 2), np.float32))
    assert silence.key is None and silence.confidence == 0.0


def test_correlation_picks_the_profile_that_matches():
    # The chroma vector of a C major triad (energy on C, E and G only).
    chroma = np.zeros(12)
    chroma[[0, 4, 7]] = 1.0
    assert key_from_chroma(chroma).key == "C major"
    assert key_from_chroma(np.ones(12)).key is None  # every note equal: no key at all


# Every key's position on the Camelot wheel (minor = A, major = B).
CAMELOT_WHEEL = {
    "Ab minor": "1A", "B major": "1B", "Eb minor": "2A", "F# major": "2B",
    "Bb minor": "3A", "Db major": "3B", "F minor": "4A", "Ab major": "4B",
    "C minor": "5A", "Eb major": "5B", "G minor": "6A", "Bb major": "6B",
    "D minor": "7A", "F major": "7B", "A minor": "8A", "C major": "8B",
    "E minor": "9A", "G major": "9B", "B minor": "10A", "D major": "10B",
    "F# minor": "11A", "A major": "11B", "C# minor": "12A", "E major": "12B",
}


@pytest.mark.parametrize("key, code", CAMELOT_WHEEL.items())
def test_camelot_codes_for_all_24_keys(key, code):
    assert camelot(key) == code


def test_camelot_accepts_sharps_and_flats():
    assert camelot("G# minor") == camelot("Ab minor") == "1A"
    assert camelot("Gb major") == camelot("F# major") == "2B"
    assert camelot(None) is None
    assert parse_key("Db major") == (1, "major")


def test_busier_brighter_louder_songs_score_higher_energy():
    calm = play([[45]] * 33, 60 / 100) * 0.5  # a soft, low note (A2) on every beat at 100 BPM
    rng = np.random.default_rng(2)
    busy = np.zeros((20 * SR, 2), np.float32)  # loud noise bursts, four per beat at 128 BPM
    burst = (0.8 * rng.standard_normal(round(0.05 * SR)) * np.exp(-np.arange(round(0.05 * SR)) / 400))
    for t in np.arange(0.1, 19.9, 60 / 128 / 4):
        i = round(t * SR)
        busy[i : i + len(burst)] += burst[:, np.newaxis].astype(np.float32)
    f_calm, f_busy = energy_features(calm), energy_features(busy)
    assert f_busy.onset_rate > 3 * f_calm.onset_rate
    assert f_busy.brightness_hz > 4 * f_calm.brightness_hz
    assert f_busy.loudness_db > f_calm.loudness_db
    low, high = energy_scores([f_calm, f_busy])
    assert low == pytest.approx(0.0) and high == pytest.approx(1.0)  # scaled across the set


def test_energy_scores_keep_near_identical_songs_close():
    a = EnergyFeatures(-12.0, 4.0, 2000.0)
    b = EnergyFeatures(-11.8, 4.1, 2050.0)  # almost the same song
    first, second = energy_scores([a, b])
    assert abs(first - second) < 0.1  # not stretched apart to 0 and 1
    assert energy_scores([a, None]) == [pytest.approx(0.5), None]  # a silent song has no energy
    assert energy_features(np.zeros((SR, 2), np.float32)) is None


def test_the_cache_keeps_key_and_energy(tmp_path):
    song = tmp_path / "song.wav"
    song.write_bytes(b"pretend audio")
    cache_file = tmp_path / "cache.json"
    saved = SongAnalysis(124.0, np.array([0.5, 0.98]), key="F# minor", key_confidence=0.7,
                         key_method="librosa", energy=EnergyFeatures(-9.5, 3.2, 2400.0))
    AnalysisCache(cache_file).put(song, saved)
    loaded = AnalysisCache(cache_file).get(song)
    assert (loaded.key, loaded.key_confidence, loaded.key_method) == ("F# minor", 0.7, "librosa")
    assert loaded.energy == EnergyFeatures(-9.5, 3.2, 2400.0)


def test_caches_from_before_key_detection_are_redone(tmp_path):
    song = tmp_path / "song.wav"
    song.write_bytes(b"pretend audio")
    cache_file = tmp_path / "cache.json"
    old = {"version": 2, "songs": {str(song.resolve()): {"mtime_ns": song.stat().st_mtime_ns, "bpm": 120.0,
                                                         "beats": [0.5, 1.0]}}}
    cache_file.write_text(json.dumps(old))
    assert analysis.CACHE_VERSION > 2
    assert AnalysisCache(cache_file).get(song) is None  # an old-version file is ignored
