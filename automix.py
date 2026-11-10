"""automix: turn a set of songs into one continuous DJ-style mix.

Stage 1: songs play in order, joined by equal-power crossfades of a fixed length.

    python automix.py songs/ -o mix.mp3 --crossfade 8
    python automix.py intro.mp3 middle.flac outro.wav -o mix.wav

The pipeline, and where each step lives:

    for each song:  load, resample to 44.1 kHz, stereo   audio_io.load_audio
                    trim silence at both ends            processing.trim_silence
                    match loudness (RMS)                 processing.normalize_loudness
    then:           crossfade the songs together         mixer.mix_tracks (+ transitions.py)
                    keep peaks below -1 dBFS             processing.clipping_gain
                    write the file                       audio_io.save_audio
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from audio_io import INPUT_EXTENSIONS, SAMPLE_RATE, check_output_path, load_audio, save_audio
from mixer import mix_tracks
from processing import CEILING_DB, TARGET_RMS_DB, normalize_loudness, rms_db, trim_silence


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crossfade songs into one continuous DJ-style mix."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="a folder of songs (mixed in alphabetical order) and/or audio files "
        "(mixed in the order given); supported: .mp3, .wav, .flac",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="mix.wav",
        help="file to write: .wav, .flac or .mp3 (MP3 needs ffmpeg). Default: mix.wav",
    )
    parser.add_argument(
        "--crossfade",
        type=float,
        default=8.0,
        metavar="SECONDS",
        help="length of each crossfade in seconds (default: 8); "
        "shortened automatically next to very short songs",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.crossfade) or args.crossfade < 0:
        parser.error("--crossfade must be a number of seconds, 0 or more")
    return args


def collect_input_files(inputs: list[str], exclude: Path | None = None) -> list[Path]:
    """Turn the command-line inputs into an ordered list of audio files.

    A folder contributes its audio files in alphabetical order, ignoring case and
    skipping hidden files (like the "._song.mp3" files macOS leaves on USB drives).
    Files named directly are kept in the order given. `exclude` is dropped from the
    list: that's the output file, so re-running into the songs folder doesn't mix
    the previous mix back in.
    """
    files: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            found = [
                p
                for p in path.iterdir()
                if p.is_file()
                and p.suffix.lower() in INPUT_EXTENSIONS
                and not p.name.startswith(".")
            ]
            files += sorted(found, key=lambda p: (p.name.casefold(), p.name))
        elif path.is_file():
            if path.suffix.lower() not in INPUT_EXTENSIONS:
                raise ValueError(
                    f"{path} isn't a supported audio file (use {', '.join(INPUT_EXTENSIONS)})"
                )
            files.append(path)
        else:
            raise FileNotFoundError(f"no such file or folder: {path}")

    if exclude is not None:
        files = [f for f in files if f.resolve() != exclude.resolve()]
    return files


def format_timestamp(seconds: float) -> str:
    """0 -> '00:00', 225.9 -> '03:45', 3725 -> '1:02:05'."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def load_tracks(paths: list[Path], sr: int = SAMPLE_RATE) -> tuple[list[np.ndarray], list[str]]:
    """Load, trim and loudness-normalize each song; return the audio and the titles.

    Files that can't be read, or that turn out to be completely silent, are
    skipped with a message rather than stopping the whole mix.
    """
    tracks, titles = [], []
    width = min(40, max((len(p.name) for p in paths), default=0))  # line the columns up
    for number, path in enumerate(paths, start=1):
        counter = f"  [{number}/{len(paths)}]"
        try:
            audio = load_audio(path, sr)
        except Exception as exc:  # corrupt or unreadable file: skip it and carry on
            print(f"{counter} {path.name}: skipped, couldn't decode it as audio ({exc})",
                  file=sys.stderr)
            continue

        audio = trim_silence(audio)  # trim first, so silence doesn't skew the loudness
        if len(audio) == 0:
            print(f"{counter} {path.name}: skipped, it's completely silent", file=sys.stderr)
            continue
        level = rms_db(audio)
        audio = normalize_loudness(audio)
        # Stage 2+: per-track analysis (analysis.estimate_bpm, detect_beats, ...) goes here.

        print(f"{counter} {path.name:<{width}}  {format_timestamp(len(audio) / sr)}  "
              f"RMS {level:6.1f} dBFS  gain {TARGET_RMS_DB - level:+5.1f} dB")
        tracks.append(audio)
        titles.append(path.stem)
    return tracks, titles


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    try:
        check_output_path(output)  # fail now, not after minutes of processing
        paths = collect_input_files(args.inputs, exclude=output)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not paths:
        print(f"error: no audio files ({', '.join(INPUT_EXTENSIONS)}) found in: "
              f"{' '.join(args.inputs)}", file=sys.stderr)
        return 1

    print(f"Preparing {len(paths)} song(s): trimming silence, "
          f"normalizing loudness to {TARGET_RMS_DB:g} dBFS RMS")
    tracks, titles = load_tracks(paths)
    if not tracks:
        print("error: none of the files could be used", file=sys.stderr)
        return 1

    crossfade = round(args.crossfade * SAMPLE_RATE)  # seconds -> samples
    print(f"\nMixing with {args.crossfade:g} s crossfades...")
    mix = mix_tracks(tracks, crossfade)
    del tracks  # the mix holds its own copy; free this memory before writing the file
    for i, actual in enumerate(mix.crossfades):
        if actual < crossfade:
            print(f"  note: {titles[i]} -> {titles[i + 1]} crossfade shortened to "
                  f"{actual / SAMPLE_RATE:.1f} s (song too short)")
    if mix.safety_gain_db < -0.05:
        print(f"  note: whole mix turned down {-mix.safety_gain_db:.1f} dB so its loudest "
              f"peak stays under {CEILING_DB:g} dBFS (see processing.prevent_clipping)")

    print(f"Writing {output}...")
    save_audio(output, mix.audio)

    print("\nTracklist")
    for number, (title, start) in enumerate(zip(titles, mix.starts), start=1):
        print(f"  {number:2d}. {format_timestamp(start / SAMPLE_RATE)}  {title}")
    print(f"\nTotal length: {format_timestamp(len(mix.audio) / SAMPLE_RATE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
