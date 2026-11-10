"""Loading and saving audio files.

Every other module works with ONE in-memory format, so none of them has to care
about file formats, sample rates or channel counts:

    a numpy array of dtype float32 and shape (n_samples, 2)

    * Row i is one instant in time; column 0 is the left speaker, column 1 the right.
    * Values run from -1.0 to +1.0 ("full scale"). Think of it as the speaker cone's
      position: 0 is at rest, +/-1 is as far as it can move.
    * There are SAMPLE_RATE rows per second of sound.

load_audio() turns any input file into that format; save_audio() turns it back
into a file.
"""

import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# 44,100 samples per second (CD quality). A sample rate can only represent
# frequencies up to half its value (the "Nyquist limit"): 22,050 Hz here, just above
# the ~20,000 Hz limit of human hearing. Every track is converted to this one rate
# because we mix by adding arrays sample by sample, which only makes sense if
# sample i means the same instant in every track.
SAMPLE_RATE = 44_100

INPUT_EXTENSIONS = (".mp3", ".wav", ".flac")
OUTPUT_EXTENSIONS = (".wav", ".flac", ".mp3")
MP3_BITRATE = "320k"  # the highest bitrate MP3 offers


def to_stereo(audio: np.ndarray) -> np.ndarray:
    """Return `audio` as a float32 array of shape (n_samples, 2).

    Accepts mono `(n,)` or `(n, 1)`, stereo `(n, 2)`, or more channels `(n, c)`.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]  # (n,) -> (n, 1)
    if audio.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D array, got shape {audio.shape}")

    channels = audio.shape[1]
    if channels == 2:
        return np.ascontiguousarray(audio)
    if channels == 1:
        # Mono -> stereo: send the same signal to both speakers.
        return np.repeat(audio, 2, axis=1)
    # Surround (e.g. 5.1): average every channel into one so no instrument is
    # dropped, then treat it like mono. Crude, but such files are rare.
    return np.repeat(audio.mean(axis=1, keepdims=True), 2, axis=1)


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Read an audio file as float32 stereo at `sr` Hz, shape (n_samples, 2)."""
    # librosa decodes the file (via soundfile) and resamples it to `sr`; mono=False
    # keeps both channels. It returns shape (n,) for mono files and (channels, n)
    # otherwise -- "channels first" -- so we transpose to put samples first.
    # (Transposing a 1-D array does nothing, so mono files pass straight through.)
    y, _ = librosa.load(path, sr=sr, mono=False)
    return to_stereo(y.T)


def check_output_path(path: str | Path) -> None:
    """Raise a helpful error if we can't write this kind of file.

    The command-line tool calls this before doing any work, so a typo or a missing
    ffmpeg shows up in a second instead of after the whole mix has been built.
    """
    ext = Path(path).suffix.lower()
    if ext not in OUTPUT_EXTENSIONS:
        raise ValueError(f"can't write '{path}': the output must end in .wav, .flac or .mp3")
    if ext == ".mp3" and shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "writing MP3 needs ffmpeg, which isn't installed or isn't on your PATH. "
            "Install it (see README.md), or write a .wav file instead."
        )


def save_audio(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Write float audio, shape (n_samples, channels), to a .wav, .flac or .mp3 file."""
    path = Path(path)
    check_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Safety net: the mixer already keeps peaks below 1.0, but this function could be
    # handed anything. Values past full scale don't fit in 16-bit integers (in the
    # MP3 conversion below they'd overflow into garbage), so clip them to +/-1.0
    # first. Clipping makes a copy, so skip it when it isn't needed.
    if audio.size and max(audio.max(), -audio.min()) > 1.0:
        audio = np.clip(audio, -1.0, 1.0)

    if path.suffix.lower() == ".mp3":
        _save_mp3(path, audio, sr)
    else:
        # soundfile converts our floats to 16-bit integers (CD quality) as it writes.
        sf.write(str(path), audio, sr, subtype="PCM_16")


def _save_mp3(path: Path, audio: np.ndarray, sr: int) -> None:
    # Imported here rather than at the top: pydub prints a warning on import when
    # ffmpeg is missing, and people writing WAV files shouldn't have to see it.
    from pydub import AudioSegment

    # The encoder takes 16-bit integers (-32768 .. 32767), not floats. We scale by
    # 32767 rather than 32768 so that +1.0 lands exactly on the largest positive
    # value instead of overflowing. Converting a block at a time means we never hold
    # an extra full-length float copy of the mix (that would be ~1.3 GB per hour).
    pcm = np.empty(audio.shape, dtype="<i2")  # "<i2" = little-endian 16-bit int
    block = 1 << 20  # ~1 million samples (~24 s) per block
    for i in range(0, len(audio), block):
        pcm[i : i + block] = np.round(audio[i : i + block] * 32767)
    channels = 1 if pcm.ndim == 1 else pcm.shape[1]
    # tobytes() writes the (n, 2) array out row by row -- L R L R L R ... -- which
    # is the "interleaved" layout audio data uses.
    segment = AudioSegment(data=pcm.tobytes(), sample_width=2, frame_rate=sr, channels=channels)
    segment.export(str(path), format="mp3", bitrate=MP3_BITRATE).close()
