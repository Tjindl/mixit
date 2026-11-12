"""Stage 4 end to end: synthetic songs with known sections, mixed at cue points.

Each song is 72 bars: pad 8, build 8, drop 16, pad 8, build 8, drop 16, pad 8
(intro, build 1, drop 1, breakdown, build 2, drop 2, outro), so its true bar lines
and section boundaries are known exactly.
"""

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import automix
import tempo
from analysis import detect_beats
from audio_io import SAMPLE_RATE as SR
from mixer import CuedTransition, _onset_near, mix_planned, plan_cued
from processing import normalize_loudness
from tempo import transition_tempo

from conftest import make_structured_song, resample_time_map

LAYOUT = (("pad", 8), ("build", 8), ("drop", 16), ("pad", 8), ("build", 8), ("drop", 16), ("pad", 8))
SONGS = (("alpha", 124.0, 57, 1), ("bravo", 126.0, 57, 2), ("charlie", 128.0, 64, 3))
LINE = re.compile(r"^  (\S+) out @ (\d+:\d\d) \((.+?)\) -> (\S+) in @ (\d+:\d\d) \((.+?)\) \| "
                  r"(blend|drop swap|echo out|crossfade) \| (perfect|good|risky)$", re.MULTILINE)


@pytest.fixture(scope="module")
def generated():
    """{name: (audio, info)} for the three songs."""
    return {name: make_structured_song(bpm, LAYOUT, lead_in_beats=2, root=root, seed=seed)
            for name, bpm, root, seed in SONGS}


@pytest.fixture
def folder(tmp_path, generated):
    path = tmp_path / "songs"
    path.mkdir()
    for name, (audio, _) in generated.items():
        sf.write(path / f"{name}.wav", audio, SR)
    return path


@pytest.fixture
def ramp_ready(monkeypatch):
    if not tempo.ramp_stretch_available():
        monkeypatch.setattr(automix, "ramp_stretch_available", lambda: True)
        monkeypatch.setattr(tempo, "stretch_time_map", resample_time_map)


def run(folder, tmp_path, capsys, *extra):
    out = tmp_path / "out" / "mix.wav"
    assert automix.main([str(folder), "-o", str(out), "--min-play", "40", *extra]) == 0
    return capsys.readouterr().out, json.loads((out.parent / "set_plan.json").read_text()), out


def seconds(clock):
    minutes, secs = clock.split(":")
    return 60 * int(minutes) + int(secs)


# Through the command line


def test_plan_only_shows_each_transition_with_its_cue_points_type_and_label(folder, tmp_path, capsys, generated,
                                                                            ramp_ready):
    stdout, plan, out = run(folder, tmp_path, capsys, "--plan-only")
    assert not out.exists()
    lines = LINE.findall(stdout)
    assert len(lines) == 2  # "alpha out @ 1:31 (end of drop 1) -> bravo in @ 0:16 (build 1) | blend | good"
    assert len(plan["transitions"]) == 2
    for (a, out_at, out_cue, b, in_at, in_cue, kind, label), t in zip(lines, plan["transitions"]):
        assert (t["from"], t["to"], t["type"], t["label"]) == (a, b, kind, label)
        assert (t["out"]["cue"], t["in"]["cue"]) == (out_cue, in_cue)
        assert abs(seconds(out_at) - t["out"]["time"]) <= 0.5 and abs(seconds(in_at) - t["in"]["time"]) <= 0.5
        # Cue points are on phrase boundaries (every 8 bars here, starting at the first bar).
        for name, time in ((a, t["out"]["time"]), (b, t["in"]["time"])):
            phrases = generated[name][1]["downbeats"][::8]
            assert np.min(np.abs(phrases - time)) < 0.03
    for song in plan["songs"]:
        assert song["plays_to"] - song["plays_from"] >= 40  # --min-play
        assert [s["label"] for s in song["sections"]] == ["intro", "build 1", "drop 1", "breakdown", "build 2",
                                                          "drop 2", "outro"]


def test_transitions_option_picks_the_technique(folder, tmp_path, capsys, ramp_ready):
    for option, kind in (("echo", "echo out"), ("dropswap", "drop swap"), ("blend", "blend")):
        stdout, plan, _ = run(folder, tmp_path, capsys, "--plan-only", "--transitions", option)
        assert {t["type"] for t in plan["transitions"]} == {kind}
        if option == "dropswap":  # from the build before a drop, straight into a drop
            assert all(t["out"]["cue"].startswith("before drop") and t["in"]["cue"].startswith("drop")
                       for t in plan["transitions"])
        if option == "blend":
            assert all(t["bass_swap_beat"] == t["beats"] // 2 for t in plan["transitions"])


def test_a_full_mix_renders_and_does_not_clip(folder, tmp_path, capsys, ramp_ready):
    stdout, plan, out = run(folder, tmp_path, capsys, "--transitions", "auto")
    audio, sr = sf.read(out)
    assert sr == SR and np.abs(audio).max() <= 1.0
    assert len(audio) / sr == pytest.approx(plan["total_seconds"], abs=0.01)


def test_cue_debug_writes_audacity_labels_and_timelines(folder, tmp_path, capsys, ramp_ready):
    stdout, plan, out = run(folder, tmp_path, capsys, "--plan-only", "--cue-debug")
    debug = out.parent / "cue_debug"
    labels = sorted(debug.glob("*.txt"))
    assert [p.name for p in labels] == [f"{i + 1:02d} {s['title']}.txt" for i, s in enumerate(plan["songs"])]
    rows = [line.split("\t") for line in labels[1].read_text().splitlines()]
    assert all(len(row) == 3 and float(row[0]) <= float(row[1]) for row in rows)  # start<TAB>end<TAB>label
    texts = [row[2] for row in rows]
    assert "drop 1" in texts and "build 2" in texts and "plays" in texts
    assert any(t.startswith("IN <- ") for t in texts) and any(t.startswith("OUT -> ") for t in texts)
    assert any(t.startswith("phrase ") for t in texts) and any(t.startswith("exit? ") for t in texts)
    if importlib.util.find_spec("matplotlib") is not None:
        assert len(list(debug.glob("*.png"))) == 3


# The techniques, measured on the rendered audio


@pytest.fixture(scope="module")
def two_songs(generated):
    """alpha and bravo, analyzed, as automix.Songs ready for rendering."""
    songs = []
    for name in ("alpha", "bravo"):
        audio, info = generated[name]
        found = detect_beats(audio)
        songs.append((automix.Song(Path(f"{name}.wav"), audio=normalize_loudness(audio), length=len(audio),
                                   bpm=found.bpm, beats=found.beats), info))
    return songs


def render(two_songs, transition, stretcher, monkeypatch):
    """Plan one transition between fresh copies of alpha and bravo, render it, and mix."""
    monkeypatch.setattr(tempo, "stretch_time_map", stretcher)
    songs = [automix.Song(s.path, audio=s.audio, length=s.length, bpm=s.bpm, beats=s.beats.copy())
             for s, _ in two_songs]
    original_beats = [s.beats.copy() for s in songs]
    plan = plan_cued([s.length for s in songs], [s.beats * SR for s in songs], [s.bpm for s in songs],
                     [transition], ramp_beats=16, fallback_crossfade=8 * SR, audio=[s.audio for s in songs])
    tracks = [tempo.apply_time_map(s.audio, rs.time_map, rs.out_start, rs.out_end, stretcher=stretcher)
              for s, rs in zip(songs, plan.songs)]
    mix_plan = automix.render_cued(songs, plan)
    mix = mix_planned([s.audio for s in songs], mix_plan, plan.transition_functions())
    return plan, tracks, mix, songs, original_beats


def test_a_drop_swap_cuts_exactly_on_the_downbeat(two_songs, stretcher, monkeypatch):
    (a, info_a), (b, info_b) = two_songs
    a_drop, b_drop = info_a["boundaries"][5], info_b["boundaries"][2]  # alpha's drop 2, bravo's drop 1
    swap = CuedTransition("dropswap", a_drop * SR, b_drop * SR, 0, transition_tempo(a.bpm, b.bpm, "middle"))
    plan, tracks, mix, songs, _ = render(two_songs, swap, stretcher, monkeypatch)
    a_drop_in_mix = mix.starts[0] + tracks[0].positions(a_drop * SR)  # where alpha's drop would have hit
    b_drop_in_mix = mix.starts[1] + tracks[1].positions(b_drop * SR)  # where bravo's drop does hit
    cut = mix.starts[1] + mix.crossfades[0]  # the end of the few-ms crossfade
    assert abs(b_drop_in_mix - a_drop_in_mix) / SR < 0.005
    assert abs(cut - a_drop_in_mix) / SR < 0.005
    assert mix.crossfades[0] / SR < 0.01  # a cut, not a blend
    # And the audio agrees: in the mix itself, the first hit of the drop starts right at the cut.
    # (Rubber Band smears a hit's very start up to ~0.5 ms earlier, into the crossfade's last moment.)
    heard = _onset_near(mix.audio, cut, SR)
    assert -0.001 <= (heard - cut) / SR < 0.003


def test_a_mid_song_blend_lines_up_the_beats_and_swaps_the_bass_on_a_downbeat(two_songs, stretcher, monkeypatch):
    (a, info_a), (b, info_b) = two_songs
    exit, entry = info_a["boundaries"][3], info_b["boundaries"][1]  # end of alpha's drop 1, bravo's build 1
    blend = CuedTransition("blend", exit * SR, entry * SR, 32, transition_tempo(a.bpm, b.bpm, "middle"))
    plan, tracks, mix, songs, _ = render(two_songs, blend, stretcher, monkeypatch)
    a_beats = mix.starts[0] + tracks[0].positions(info_a["beats"] * SR)
    b_beats = mix.starts[1] + tracks[1].positions(info_b["beats"] * SR)
    overlap = (b_beats >= mix.starts[1] - 100) & (b_beats < mix.starts[1] + mix.crossfades[0] - 100)
    assert overlap.sum() == 32
    gaps_ms = [np.min(np.abs(a_beats - t)) / SR * 1000 for t in b_beats[overlap]]
    assert max(gaps_ms) < 10
    swap_in_mix = mix.starts[1] + plan.swaps[0]
    a_bars = mix.starts[0] + tracks[0].positions(info_a["downbeats"] * SR)
    b_bars = mix.starts[1] + tracks[1].positions(info_b["downbeats"] * SR)
    assert np.min(np.abs(a_bars - swap_in_mix)) / SR < 0.01  # the swap is on a downbeat of both songs
    assert np.min(np.abs(b_bars - swap_in_mix)) / SR < 0.01


def test_an_echo_out_decays_away_without_clipping(two_songs, monkeypatch):
    (a, info_a), (b, info_b) = two_songs
    exit, entry = info_a["boundaries"][3], info_b["boundaries"][2]  # end of alpha's drop 1, into bravo's drop 1
    echo = CuedTransition("echo", exit * SR, entry * SR, 8)
    plan, tracks, mix, songs, _ = render(two_songs, echo, resample_time_map, monkeypatch)
    tail = songs[0].audio[-plan.overlaps[0] :]  # alpha's track ends with the echo tail
    beat = plan.echo_delays[0]
    levels = [np.sqrt(np.mean(tail[k * beat : (k + 1) * beat] ** 2)) for k in range(8)]
    assert all(later < earlier for earlier, later in zip(levels, levels[1:]))  # every repeat quieter
    assert levels[-1] < 0.05 * levels[0]
    assert np.abs(mix.audio).max() <= 1.0
    assert plan.songs[0].ramps == () and plan.songs[1].ramps == ()  # no tempo change for an echo
