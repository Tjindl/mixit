"""Shared test helpers. The tests use generated audio, so no real songs are needed."""

import numpy as np
import pytest

import analysis
import tempo
from audio_io import SAMPLE_RATE


def make_sine(freq=440.0, seconds=1.0, amplitude=0.5, sr=SAMPLE_RATE, stereo=True):
    """A pure tone: float32, shape (n, 2), or (n,) if stereo=False."""
    t = np.arange(round(seconds * sr)) / sr  # the time, in seconds, of each sample
    wave = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.column_stack([wave, wave]) if stereo else wave


def make_click_track(bpm, seconds=30.0, offset=0.3, sr=SAMPLE_RATE, seed=0):
    """A synthetic 'song' with a drum-like hit on every beat at exactly `bpm`.

    Each hit is a 55 Hz thump (like a kick drum) with a 4 ms burst of noise at the
    start (the sharp attack that beat detection keys on). Returns (audio, beat_times):
    stereo float32 audio, and the true time of every hit in seconds.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(round(0.08 * sr)) / sr
    hit = 0.8 * np.sin(2 * np.pi * 55 * t) * np.exp(-t / 0.03)
    attack = round(0.004 * sr)
    hit[:attack] += 0.6 * rng.standard_normal(attack)

    beat_times = np.arange(offset, seconds - 0.1, 60.0 / bpm)
    mono = np.zeros(round(seconds * sr))
    for time in beat_times:
        i = round(time * sr)
        mono[i : i + len(hit)] += hit[: len(mono) - i]
    mono = mono.astype(np.float32)
    return np.column_stack([mono, mono]), beat_times


def make_syncopated_track(bpm, seconds=40.0, seed=0, sr=SAMPLE_RATE):
    """A kick on every beat, plus loud noise 'stabs' on random off-beat 16th notes.

    The stabs are louder than the kicks, which tempts a loosely held beat tracker to
    wander off the grid toward them. Returns (audio, beat_times) like make_click_track.
    """
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    t = np.arange(round(0.1 * sr)) / sr
    kick = 0.7 * np.sin(2 * np.pi * 55 * t) * np.exp(-t / 0.03)
    kick[:150] += 0.4 * rng.standard_normal(150)
    stab = 1.2 * rng.standard_normal(len(t)) * np.exp(-t / 0.02)
    mono = np.zeros(round(seconds * sr))

    def add(sound, time):
        i = round(time * sr)
        if i < len(mono):  # a stab after the last beat may fall past the end
            mono[i : i + len(sound)] += sound[: len(mono) - i]

    beat_times = np.arange(0.3, seconds - 0.2, beat)
    for time in beat_times:
        add(kick, time)
        for sixteenth in (1, 2, 3):
            if rng.random() < 0.35:
                add(stab, time + sixteenth * beat / 4)
    mono = mono.astype(np.float32)
    return np.column_stack([mono, mono]), beat_times


def make_structured_song(bpm=126.0, sections=(("pad", 8), ("drop", 8), ("pad", 8), ("drop", 8)),
                         lead_in_beats=1, root=57, seed=0, sr=SAMPLE_RATE):
    """A synthetic dance track built from known sections, each a number of 4-beat bars.

    Section kinds:
      pad        quiet: a soft chord and a faint hi-hat on every beat (intros, breakdowns)
      drop       loud: kick on every beat (beat 1 accented), a bass line, off-beat hats, chord
      build      rising: a snare that doubles in density every 2 bars, and a noise riser
    The song starts `lead_in_beats` beats before the first bar line (so the first beat
    is NOT a downbeat), with a soft hat on those beats. Returns (audio, info): stereo
    float32, and a dict with the true "beats", "downbeats" and section "boundaries"
    (all in seconds), plus the section "kinds".
    """
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    n_bars = sum(bars for _, bars in sections)
    first_bar = 0.25 + lead_in_beats * beat  # time of the first bar line
    length = round((first_bar + n_bars * 4 * beat + 1.0) * sr)
    mono = np.zeros(length)

    def add(sound, time, gain=1.0):
        i = round(time * sr)
        if 0 <= i < length:
            mono[i : i + len(sound)] += gain * sound[: length - i]

    def tone(freq, seconds, decay, harmonics=(1,)):
        t = np.arange(round(seconds * sr)) / sr
        env = np.minimum(1.0, t / 0.005) * np.exp(-t / decay)
        return env * sum(np.sin(2 * np.pi * freq * h * t) / h for h in harmonics)

    def noise(seconds, decay):
        n = round(seconds * sr)
        return rng.standard_normal(n) * np.exp(-np.arange(n) / (decay * sr))

    kick = 0.9 * tone(52.0, 0.25, 0.06)
    kick[:120] += 0.3 * rng.standard_normal(120)
    hat = 0.15 * np.diff(noise(0.04, 0.008), prepend=0)  # a click of bright noise
    snare = 0.35 * (noise(0.15, 0.04) + tone(190.0, 0.15, 0.04))
    freqs = [440.0 * 2 ** ((m - 69) / 12) for m in (root, root + 3, root + 7)]  # a minor chord

    beats, downbeats, boundaries, kinds = [], [], [], []
    for k in range(lead_in_beats):
        t = first_bar - (lead_in_beats - k) * beat
        beats.append(t)
        add(hat, t, 0.5)
    bar = 0
    for kind, bars in sections:
        start = first_bar + bar * 4 * beat
        boundaries.append(start)
        kinds.append(kind)
        for b in range(bars):
            bar_start = first_bar + (bar + b) * 4 * beat
            downbeats.append(bar_start)
            add(sum(tone(f, 4 * beat, 4 * beat, (1, 2, 3)) for f in freqs), bar_start,
                0.05 if kind != "drop" else 0.08)
            for i in range(4):
                t = bar_start + i * beat
                beats.append(t)
                if kind == "drop":
                    add(kick, t, 1.0 if i == 0 else 0.55)  # beat 1 accented
                    add(hat, t + beat / 2, 1.0)
                    add(0.35 * tone(freqs[0] / 4, beat / 2, beat), t)  # bass, 8th notes
                    add(0.35 * tone(freqs[0] / 4 * (1.5 if i % 2 else 1.0), beat / 2, beat), t + beat / 2)
                elif kind == "build":
                    hits = 2 ** min(2, b // 2)  # 1, 2, 4 snares per beat as the build goes on
                    for h in range(hits):
                        add(snare, t + h * beat / hits, 0.4 + 0.6 * b / bars)
                    add(hat, t, 0.6)
                else:  # pad
                    add(hat, t, 0.6)
        if kind == "build":  # a noise riser across the whole build
            n = round(bars * 4 * beat * sr)
            riser = rng.standard_normal(n) * np.linspace(0.0, 0.25, n) ** 2 * 4
            add(riser, start)
        bar += bars
    boundaries.append(first_bar + n_bars * 4 * beat)
    mono = (0.5 * mono / np.max(np.abs(mono))).astype(np.float32)
    info = {"beats": np.array(beats), "downbeats": np.array(downbeats),
            "boundaries": np.array(boundaries), "kinds": kinds, "beat": beat}
    return np.column_stack([mono, mono]), info


def find_hit_centroids(audio, approx_positions, sr=SAMPLE_RATE):
    """Where each hit's energy is centred (in samples), searched near its expected position.

    The centre of energy, rather than the loudest sample, gives a stable answer
    even when time-stretching has smeared a hit into a blur.
    """
    energy = np.sum(np.square(audio, dtype=np.float64), axis=1)
    centroids = []
    for p in approx_positions:
        a, b = max(0, int(p - 0.04 * sr)), min(len(energy), int(p + 0.08 * sr))
        e = energy[a:b]
        centroids.append(a + np.sum(np.arange(len(e)) * e) / np.sum(e))
    return np.array(centroids)


def resample_time_map(audio, src, dst, sr=SAMPLE_RATE):
    """A stand-in for Rubber Band: plays `audio` faster or slower along a time map.

    Like a turntable, it changes the pitch along with the speed, so it's no good for
    music. But it follows the time map exactly and copies rate-1 stretches sample
    for sample, which is all the timing tests need, and it works without Rubber Band.
    """
    positions = np.interp(np.arange(dst[-1]), dst, src)  # the input position of each output sample
    index = np.arange(len(audio))
    return np.column_stack([np.interp(positions, index, audio[:, c]) for c in range(audio.shape[1])]
                           ).astype(np.float32)


@pytest.fixture(params=["resample", "rubberband"])
def stretcher(request):
    """Each ramp test runs with the resampling stand-in, and again with Rubber Band if it's installed."""
    if request.param == "rubberband":
        if not tempo.ramp_stretch_available():
            pytest.skip("Rubber Band 3.0+ isn't installed")
        return tempo.stretch_time_map
    return resample_time_map


@pytest.fixture
def syncopated_track():
    """Gives a test the make_syncopated_track function."""
    return make_syncopated_track


@pytest.fixture
def hit_centroids():
    """Gives a test the find_hit_centroids function."""
    return find_hit_centroids


@pytest.fixture
def sine():
    """Gives a test the make_sine function: `sine(440, seconds=2, amplitude=0.3)`."""
    return make_sine


@pytest.fixture
def click_track():
    """Gives a test the make_click_track function: `audio, beats = click_track(120)`."""
    return make_click_track


@pytest.fixture(autouse=True)
def _isolated_analysis_cache(tmp_path, monkeypatch):
    """Point the analysis cache at a temporary file, so tests never touch the real one."""
    monkeypatch.setattr(analysis, "DEFAULT_CACHE_PATH", tmp_path / "analysis_cache.json")


@pytest.fixture(autouse=True)
def _without_neural_networks(request, monkeypatch):
    """Tests use our own beat tracking and sections, even if beat_this, demucs or allin1 is installed.

    Those run neural networks (seconds to minutes per song), and the tests analyze
    many songs. tests/test_optional_tools.py tests them on their own, when installed.
    """
    if request.node.get_closest_marker("neural"):
        return
    import automix
    import structure

    for module, name in ((analysis, "beat_this_available"), (automix, "beat_this_available"),
                         (structure, "demucs_available"), (automix, "demucs_available"),
                         (structure, "allin1_available")):
        monkeypatch.setattr(module, name, lambda: False)
