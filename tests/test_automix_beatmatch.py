"""End-to-end tests of the Stage 2 command-line options, on generated click tracks.

Stage 2's one-tempo-for-everything mix is now --tempo-mode global (Stage 3 made
ramp mode the default), so the tests of its output ask for it by name, and those
that depend on the order of the songs ask for --order given.
"""

import re

import numpy as np
import pytest
import soundfile as sf

import analysis
import automix
from audio_io import SAMPLE_RATE, load_audio
from processing import trim_silence


def tracklist(stdout):
    """Parse the tracklist's song rows into (timestamp, title, BPM) tuples."""
    section = stdout.split("Tracklist")[-1]
    return re.findall(r"^\s*\d+\.\s+(\S+)\s+(\S+)\s+([\d.]+|-)\s", section, flags=re.MULTILINE)


def make_folder(path, click_track, bpms):
    path.mkdir()
    for name, bpm in bpms:
        audio, _ = click_track(bpm, seconds=24, seed=bpm)
        sf.write(path / f"{name}.wav", audio, SAMPLE_RATE)
    return path


@pytest.fixture
def click_songs(tmp_path, click_track):
    """Two click-track 'songs' at 120 and 126 BPM."""
    return make_folder(tmp_path / "songs", click_track, [("a_120", 120), ("b_126", 126)])


def test_beatmatched_mix_end_to_end(click_songs, tmp_path, capsys):
    out = tmp_path / "mix.wav"
    assert automix.main([str(click_songs), "-o", str(out), "--tempo-mode", "global", "--order", "given"]) == 0
    stdout = capsys.readouterr().out
    assert "mixing at 123.0 BPM (the median of the songs)" in stdout
    assert "a_120 -> b_126: beat-matched, 16 beats" in stdout
    rows = tracklist(stdout)
    assert [(title, bpm) for _, title, bpm in rows] == [("a_120", "120.0"), ("b_126", "126.0")]
    assert "-> 123.0 BPM: a_120 +2.5%, b_126 -2.4%" in stdout  # each song's stretch
    audio, sr = sf.read(out)
    assert sr == SAMPLE_RATE and audio.shape[1] == 2
    assert np.abs(audio).max() <= 1.0
    assert 35 < len(audio) / sr < 45  # two ~23 s songs overlapping by ~8 s


def test_analyze_only_prints_a_table_and_writes_nothing(click_songs, tmp_path, capsys):
    out = tmp_path / "mix.wav"
    assert automix.main([str(click_songs), "--analyze-only", "-o", str(out), "--tempo-mode", "global"]) == 0
    assert not out.exists()
    stdout = capsys.readouterr().out
    assert "Mix tempo: 123.0 BPM" in stdout
    rows = re.findall(r"^\s+\d+\s+(\S+)\s+([\d.]+)\s+(\d+)\s+([+-][\d.]+%)", stdout, flags=re.MULTILINE)
    assert [(name, bpm, stretch) for name, bpm, _, stretch in rows] == [
        ("a_120", "120.0", "+2.5%"),
        ("b_126", "126.0", "-2.4%"),
    ]
    assert all(int(beats) > 40 for _, _, beats, _ in rows)


def test_bpm_option_sets_the_mix_tempo(click_songs, capsys):
    assert automix.main([str(click_songs), "--analyze-only", "--bpm", "124"]) == 0
    stdout = capsys.readouterr().out
    assert "Mix tempo: 124.0 BPM (set with --bpm)" in stdout
    assert "+3.3%" in stdout and "-1.6%" in stdout


def test_analysis_is_cached_between_runs(click_songs, monkeypatch, capsys):
    assert automix.main([str(click_songs), "--analyze-only"]) == 0
    assert analysis.DEFAULT_CACHE_PATH.exists()

    def must_not_run(*args, **kwargs):
        raise AssertionError("beat detection ran again instead of using the cache")

    monkeypatch.setattr(automix, "detect_beats", must_not_run)
    assert automix.main([str(click_songs), "--analyze-only"]) == 0
    assert capsys.readouterr().out.count("reused from the cache") == 2


def test_no_beatmatch_gives_the_stage_1_mix(click_songs, tmp_path, capsys):
    out = tmp_path / "mix.wav"
    assert automix.main([str(click_songs), "-o", str(out), "--no-beatmatch", "--crossfade", "2"]) == 0
    stdout = capsys.readouterr().out
    assert "BPM" not in stdout
    trimmed = [len(trim_silence(load_audio(p))) for p in sorted(click_songs.glob("*.wav"))]
    audio, _ = sf.read(out)
    assert len(audio) == sum(trimmed) - 2 * SAMPLE_RATE  # Stage 1: whole songs, 2 s crossfade


def test_a_song_too_far_off_tempo_gets_plain_crossfades(tmp_path, click_track, capsys):
    folder = make_folder(tmp_path / "songs", click_track, [("a_120", 120), ("b_121", 121), ("c_140", 140)])
    assert automix.main([str(folder), "-o", str(tmp_path / "mix.wav"), "--tempo-mode", "global",
                         "--order", "given"]) == 0
    captured = capsys.readouterr()
    assert "c_140 is 140.0 BPM and would need a -13.6% stretch (limit 8%)" in captured.err
    assert "a_120 -> b_121: beat-matched" in captured.out
    assert "b_121 -> c_140: plain 8.0 s crossfade (c_140 isn't tempo-matched)" in captured.out
    assert tracklist(captured.out)[2][1:] == ("c_140", "140.0")
    assert "-> plain 8.0 s crossfade (c_140 isn't tempo-matched)" in captured.out.split("Tracklist")[-1]


def test_clicks_option_adds_clicks(click_songs, tmp_path):
    plain, clicked = tmp_path / "plain.wav", tmp_path / "clicked.wav"
    assert automix.main([str(click_songs), "-o", str(plain), "--tempo-mode", "global"]) == 0
    assert automix.main([str(click_songs), "-o", str(clicked), "--clicks", "--tempo-mode", "global"]) == 0

    def energy_near(path, freq):  # how much sound there is around `freq` Hz
        audio, sr = sf.read(path)
        spectrum = np.abs(np.fft.rfft(audio[:, 0]))
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)
        return np.sum(spectrum[(freqs > freq - 50) & (freqs < freq + 50)] ** 2)

    assert energy_near(clicked, 1500) > 4 * energy_near(plain, 1500)  # odd songs' click pitch
    assert energy_near(clicked, 2500) > 4 * energy_near(plain, 2500)  # even songs' click pitch


@pytest.mark.parametrize("bad", [["--crossfade-beats", "0"], ["--bpm", "-5"], ["--bpm", "nan"],
                                 ["--max-stretch", "-1"]])
def test_bad_options_are_rejected(tmp_path, bad):
    with pytest.raises(SystemExit):
        automix.main([str(tmp_path), *bad])
