"""The command line's Stage 3 options: --order, --start, --energy, --plan-only and set_plan.json."""

import json
import re

import pytest
import soundfile as sf

import automix
import tempo
from audio_io import SAMPLE_RATE as SR

from conftest import make_click_track, resample_time_map


def write_songs(folder, bpms, seconds=24):
    folder.mkdir()
    for bpm in bpms:
        sf.write(folder / f"s{bpm}.wav", make_click_track(bpm, seconds=seconds, seed=bpm)[0], SR)
    return folder


def song_rows(stdout):
    """The tracklist's song rows: (position, start, title, BPM, key, energy)."""
    section = stdout.split("Tracklist")[-1]
    return re.findall(r"^\s*(\d+)\.\s+(\S+)\s+(\S+)\s+([\d.]+|-)\s+(\S+)\s+([\d.]+|-)$", section, flags=re.MULTILINE)


@pytest.fixture
def ramp_ready(monkeypatch):
    """Ramp mode, whether or not Rubber Band is installed (the resampling stand-in fills in)."""
    if not tempo.ramp_stretch_available():
        monkeypatch.setattr(automix, "ramp_stretch_available", lambda: True)
        monkeypatch.setattr(tempo, "stretch_time_map", resample_time_map)


def test_plan_only_prints_the_set_saves_the_plan_and_renders_nothing(tmp_path, ramp_ready, capsys):
    folder = write_songs(tmp_path / "songs", (130, 120, 125))
    out = tmp_path / "out" / "mix.wav"
    assert automix.main([str(folder), "-o", str(out), "--plan-only"]) == 0
    assert not out.exists()
    stdout = capsys.readouterr().out

    rows = song_rows(stdout)
    assert [title for _, _, title, *_ in rows] in (["s120", "s125", "s130"], ["s130", "s125", "s120"])
    assert [start for _, start, *_ in rows][0] == "00:00"
    assert all(re.fullmatch(r"\d+[AB]\??", key) for *_, key, _ in rows)  # Camelot keys
    transitions = re.findall(r"-> ([\d.]+) BPM: .* \[cost ([\d.]+), (perfect|good|risky)\]", stdout)
    assert len(transitions) == 2

    plan = json.loads((tmp_path / "out" / "set_plan.json").read_text())
    assert [s["title"] for s in plan["songs"]] == [title for _, _, title, *_ in rows]
    assert [s["position"] for s in plan["songs"]] == [1, 2, 3]
    assert plan["settings"]["tempo_mode"] == "ramp" and plan["ordering"]["method"] == "brute force"
    first = plan["transitions"][0]
    assert first["type"] == "beat-matched" and first["label"] in ("perfect", "good", "risky")
    assert first["tempo_bpm"] == pytest.approx(float(transitions[0][0]), abs=0.05)
    a, b = plan["songs"][0]["bpm"], plan["songs"][1]["bpm"]
    assert first["tempo_bpm"] == pytest.approx((a * b) ** 0.5, abs=0.05)  # met in the middle
    assert set(plan["songs"][0]) >= {"start_seconds", "bpm", "key", "camelot", "key_confidence", "energy"}


def test_the_plan_is_saved_after_a_real_mix_too(tmp_path, ramp_ready, capsys):
    folder = write_songs(tmp_path / "songs", (120, 126))
    out = tmp_path / "mix.wav"
    assert automix.main([str(folder), "-o", str(out)]) == 0
    plan = json.loads((tmp_path / "set_plan.json").read_text())
    starts = [s["start_seconds"] for s in plan["songs"]]
    assert starts[0] == 0 and 10 < starts[1] < 20  # where the second song came in
    assert plan["total_seconds"] == pytest.approx(len(sf.read(out)[0]) / SR, abs=0.01)


def test_order_given_and_alpha(tmp_path, capsys):
    folder = write_songs(tmp_path / "songs", (126, 120))
    files = [str(folder / "s126.wav"), str(folder / "s120.wav")]
    assert automix.main([*files, "--plan-only", "--order", "given", "-o", str(tmp_path / "m.wav")]) == 0
    assert [r[2] for r in song_rows(capsys.readouterr().out)] == ["s126", "s120"]
    assert automix.main([*files, "--plan-only", "--order", "alpha", "-o", str(tmp_path / "m.wav")]) == 0
    assert [r[2] for r in song_rows(capsys.readouterr().out)] == ["s120", "s126"]


@pytest.mark.parametrize("order", ["smart", "given"])
def test_start_opens_the_set(tmp_path, ramp_ready, capsys, order):
    folder = write_songs(tmp_path / "songs", (120, 125, 130))
    assert automix.main([str(folder), "--plan-only", "--order", order, "--start", "s125",
                         "-o", str(tmp_path / "m.wav")]) == 0
    stdout = capsys.readouterr().out
    assert song_rows(stdout)[0][2] == "s125"
    assert "opening with s125" in stdout


def test_an_unknown_start_song_is_a_clear_error(tmp_path, capsys):
    folder = write_songs(tmp_path / "songs", (120, 125))
    assert automix.main([str(folder), "--plan-only", "--start", "nope.mp3"]) == 1
    assert "--start: no song called 'nope.mp3'" in capsys.readouterr().err


def test_energy_shape_is_reported_and_saved(tmp_path, ramp_ready, capsys):
    folder = write_songs(tmp_path / "songs", (120, 125, 130))
    assert automix.main([str(folder), "--plan-only", "--energy", "rise", "-o", str(tmp_path / "m.wav")]) == 0
    assert "energy shape 'rise'" in capsys.readouterr().out
    assert json.loads((tmp_path / "set_plan.json").read_text())["settings"]["energy"] == "rise"


def test_global_mode_plan_shows_the_mix_tempo(tmp_path, capsys):
    folder = write_songs(tmp_path / "songs", (120, 126))
    assert automix.main([str(folder), "--plan-only", "--tempo-mode", "global", "--order", "given",
                         "-o", str(tmp_path / "m.wav")]) == 0
    stdout = capsys.readouterr().out
    assert "mixed at 123.0 BPM" in stdout
    assert "-> 123.0 BPM: s120 +2.5%, s126 -2.4%" in stdout


def test_analyze_only_shows_key_and_energy(tmp_path, capsys):
    folder = write_songs(tmp_path / "songs", (120, 126))
    assert automix.main([str(folder), "--analyze-only"]) == 0
    stdout = capsys.readouterr().out
    assert "Ramp mode: each song keeps its own tempo" in stdout
    rows = re.findall(r"^\s+\d+\s+(s\d+)\s+([\d.]+)\s+(\d+)\s+(\d+[AB]) (\S+ (?:major|minor))\s+([\d.]+)\s+([\d.]+)",
                      stdout, flags=re.MULTILINE)
    assert [r[0] for r in rows] == ["s120", "s126"]
