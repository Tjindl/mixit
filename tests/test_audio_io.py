import shutil

import numpy as np
import pytest
import soundfile as sf

from audio_io import SAMPLE_RATE, check_output_path, load_audio, save_audio, to_stereo


def test_to_stereo_handles_every_channel_layout():
    mono = np.arange(5, dtype=np.float32)
    for layout in (mono, mono[:, np.newaxis]):  # (n,) and (n, 1)
        np.testing.assert_array_equal(to_stereo(layout), np.column_stack([mono, mono]))
    stereo = np.column_stack([mono, -mono])
    np.testing.assert_array_equal(to_stereo(stereo), stereo)
    assert to_stereo(np.ones((5, 6))).shape == (5, 2)  # 5.1 surround


def test_load_resamples_to_44k_and_makes_mono_stereo(tmp_path, sine):
    path = tmp_path / "mono_22k.wav"
    sf.write(path, sine(440, 1.0, sr=22_050, stereo=False), 22_050)
    audio = load_audio(path)
    assert audio.shape == (SAMPLE_RATE, 2) and audio.dtype == np.float32
    np.testing.assert_array_equal(audio[:, 0], audio[:, 1])


@pytest.mark.parametrize("name", ["out.wav", "out.flac"])
def test_save_then_load_round_trip(tmp_path, sine, name):
    original = sine(440, 0.5, amplitude=0.5)
    save_audio(tmp_path / name, original)
    # 16-bit files store 65,536 levels, so allow for that much rounding.
    np.testing.assert_allclose(load_audio(tmp_path / name), original, atol=1e-4)


def test_unknown_output_format_is_rejected():
    with pytest.raises(ValueError):
        check_output_path("mix.ogg")


@pytest.mark.skipif("MP3" not in sf.available_formats(), reason="this libsndfile can't write MP3")
def test_loads_mp3_files(tmp_path, sine):
    path = tmp_path / "song.mp3"
    sf.write(path, sine(440, 1.0), SAMPLE_RATE)  # make a test MP3 without needing ffmpeg
    audio = load_audio(path)
    assert audio.shape[1] == 2
    assert abs(len(audio) - SAMPLE_RATE) < 0.1 * SAMPLE_RATE  # MP3 may add a little padding


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="MP3 export needs ffmpeg")
def test_mp3_export(tmp_path, sine):
    path = tmp_path / "mix.mp3"
    save_audio(path, sine(440, 2.0, amplitude=0.5))
    audio = load_audio(path)
    assert audio.shape[1] == 2
    assert abs(len(audio) - 2 * SAMPLE_RATE) < 0.1 * SAMPLE_RATE
    assert np.abs(audio).max() <= 1.0
