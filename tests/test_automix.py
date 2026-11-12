"""End-to-end tests: generated songs on disk -> command-line tool -> mix file.

Smart ordering is the default since Stage 3, so tests about the order of the
inputs ask for --order given.
"""

import re

import numpy as np
import pytest
import soundfile as sf

import automix
from audio_io import SAMPLE_RATE


def tracklist(stdout):
    """Parse the printed tracklist into (timestamp, title) pairs."""
    section = stdout.split("Tracklist")[-1]
    return re.findall(r"^\s*\d+\.\s+(\S+)\s+(\S+)", section, flags=re.MULTILINE)


@pytest.fixture
def songs(tmp_path, sine):
    """A folder of three short 'songs' in different formats, sample rates and channel counts."""
    folder = tmp_path / "songs"
    folder.mkdir()
    sf.write(folder / "b_track.wav", sine(440, 2.0), 44_100)  # stereo, 44.1 kHz
    sf.write(folder / "a_track.flac", sine(330, 3.0, sr=22_050, stereo=False), 22_050)  # mono
    sf.write(folder / "C_track.wav", sine(550, 2.0, sr=48_000), 48_000)  # capital C: case is ignored
    (folder / "notes.txt").write_text("not audio, should be ignored")
    return folder


def test_mixes_a_folder_end_to_end(songs, tmp_path, capsys):
    out = tmp_path / "mix.wav"
    assert automix.main([str(songs), "-o", str(out), "--crossfade", "1", "--order", "given"]) == 0

    audio, sr = sf.read(out)
    assert sr == SAMPLE_RATE and audio.shape[1] == 2
    assert len(audio) == 5 * SAMPLE_RATE  # 3 s + 2 s + 2 s, minus two 1 s crossfades
    assert np.abs(audio).max() <= 1.0
    # Alphabetical order (ignoring case), each song starting 1 s before the last one ends:
    assert tracklist(capsys.readouterr().out) == [
        ("00:00", "a_track"),
        ("00:02", "b_track"),
        ("00:03", "C_track"),
    ]


def test_files_named_on_the_command_line_keep_their_order(songs, tmp_path, capsys):
    args = [str(songs / "C_track.wav"), str(songs / "a_track.flac"), "-o", str(tmp_path / "mix.wav"),
            "--order", "given"]
    assert automix.main(args) == 0
    assert [title for _, title in tracklist(capsys.readouterr().out)] == ["C_track", "a_track"]


def test_a_previous_mix_in_the_songs_folder_is_not_mixed_in(songs, capsys):
    out = songs / "mix.wav"
    automix.main([str(songs), "-o", str(out), "--order", "given"])
    automix.main([str(songs), "-o", str(out), "--order", "given"])  # second run: mix.wav is now in the folder
    titles = [title for _, title in tracklist(capsys.readouterr().out)]
    assert titles == ["a_track", "b_track", "C_track"]


def test_an_empty_folder_is_a_clear_error(tmp_path, capsys):
    assert automix.main([str(tmp_path), "-o", str(tmp_path / "mix.wav")]) == 1
    assert "no audio files" in capsys.readouterr().err


def test_format_timestamp():
    assert automix.format_timestamp(0) == "00:00"
    assert automix.format_timestamp(225.9) == "03:45"
    assert automix.format_timestamp(3725) == "1:02:05"
