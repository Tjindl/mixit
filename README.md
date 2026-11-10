# mixit: auto-DJ

Turns a set of songs into one continuous DJ-style mix.

**Stage 1 (this version):** songs play in order, joined by equal-power crossfades
of a fixed length. There's no beat matching yet; see the [roadmap](#roadmap).

```bash
python automix.py songs/ -o mix.mp3 --crossfade 8
```

Before mixing, every song is:

- loaded (mp3, wav or flac) and converted to 44.1 kHz stereo
- trimmed of silence at the start and end
- loudness-normalized (RMS), so every song plays at about the same volume

The songs are then crossfaded together, the result is kept from clipping, and the
mix is written to a `.wav`, `.flac` or `.mp3` file along with a printed tracklist.

## Setup

You need **Python 3.10 or newer**. Check with `python3 --version` (Windows:
`py --version`).

### 1. Create a virtual environment and install the packages

A virtual environment ("venv") is a private folder of packages for this project,
so they don't clash with anything else on your computer.

macOS / Linux:

```bash
cd mixit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
cd mixit
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell refuses to run the activate script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. In Command Prompt, use
`.venv\Scripts\activate.bat` instead.

Activate the venv again in every new terminal before using the tool. Your prompt
shows `(.venv)` when it's active.

### 2. Install ffmpeg (only needed for MP3 output)

Writing MP3 goes through pydub, which runs the separate `ffmpeg` program to do the
encoding. Reading MP3 files, and writing WAV or FLAC, work without it.

| System        | Command                                                         |
| ------------- | --------------------------------------------------------------- |
| macOS         | `brew install ffmpeg` (needs [Homebrew](https://brew.sh))       |
| Windows       | `winget install --id Gyan.FFmpeg -e`, then **open a new terminal** |
| Ubuntu/Debian | `sudo apt update && sudo apt install ffmpeg`                    |
| Fedora        | `sudo dnf install ffmpeg-free` (or `ffmpeg` from RPM Fusion)    |
| Arch          | `sudo pacman -S ffmpeg`                                         |

On Windows, `choco install ffmpeg` or `scoop install ffmpeg` work too. Check that
it's installed with `ffmpeg -version`.

## Usage

```bash
# every song in a folder, in alphabetical order
python automix.py songs/ -o mix.mp3 --crossfade 8

# particular songs, in the order you list them
python automix.py intro.mp3 peak.flac outro.wav -o mix.wav

# no ffmpeg yet? write a WAV (mix.wav is the default output)
python automix.py songs/
```

| Option                | Meaning                                                                                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `INPUT ...`           | Folders and/or files. A folder's `.mp3`/`.wav`/`.flac` files are used in alphabetical order (ignoring case and skipping hidden files); files are used in the order given. |
| `-o`, `--output`      | The mix to write: `.wav`, `.flac` (16-bit) or `.mp3` (320 kbps, needs ffmpeg). Default: `mix.wav`.                                                           |
| `--crossfade SECONDS` | Length of each crossfade. Default: 8. Shortened automatically next to a song that's too short for it.                                                         |

To control the order of a folder, start the file names with numbers (`01 ...`,
`02 ...`).

Example run, on four generated test songs:

```text
Preparing 4 song(s): trimming silence, normalizing loudness to -16 dBFS RMS
  [1/4] 01 Sunrise.mp3          00:20  RMS  -37.8 dBFS  gain +21.8 dB
  [2/4] 02 Midnight.flac        00:24  RMS  -13.2 dBFS  gain  -2.8 dB
  [3/4] 03 Short Interlude.wav  00:05  RMS  -22.2 dBFS  gain  +6.2 dB
  [4/4] 04 Finale.wav           00:20  RMS  -28.4 dBFS  gain +12.4 dB

Mixing with 8 s crossfades...
  note: 02 Midnight -> 03 Short Interlude crossfade shortened to 2.5 s (song too short)
  note: 03 Short Interlude -> 04 Finale crossfade shortened to 2.5 s (song too short)
  note: whole mix turned down 12.6 dB so its loudest peak stays under -1 dBFS (see processing.prevent_clipping)
Writing mix.mp3...

Tracklist
   1. 00:00  01 Sunrise
   2. 00:12  02 Midnight
   3. 00:33  03 Short Interlude
   4. 00:36  04 Finale

Total length: 00:56
```

The lengths shown while preparing are after silence trimming. Tracklist times are
when each song *starts*, meaning when it begins fading in.

## How it works

```text
for each song:  load, resample to 44.1 kHz, stereo   audio_io.load_audio
                trim silence at both ends            processing.trim_silence
                match loudness (RMS)                 processing.normalize_loudness
then:           crossfade the songs together         mixer.mix_tracks + transitions.py
                keep peaks below -1 dBFS             processing.clipping_gain
                write the file                       audio_io.save_audio
```

The comments in the code explain *why* each step works the way it does. Good
places to start:

- **Equal-power crossfades** (why sine and cosine): the comment block in `transitions.py`
- **RMS and decibels**: the top of `processing.py`, and `processing.rms`
- **Trimming silence** (and why it must happen before measuring loudness):
  `processing.trim_silence` and `normalize_loudness`
- **How tracks are laid out, and why crossfades shrink for short songs**: the
  diagram at the top of `mixer.py`, and `plan_crossfades`
- **Clipping, and the -1 dB ceiling**: `processing.prevent_clipping`

The processing functions are pure numpy (an array goes in, a new array comes out),
so you can experiment with them in a Python shell:

```python
from audio_io import load_audio
from processing import rms_db, trim_silence

song = load_audio("songs/01 Sunrise.mp3")  # float32 array, shape (samples, 2)
print(song.shape, rms_db(song), rms_db(trim_silence(song)))
```

Settings you might want to experiment with are constants at the top of
`processing.py`: `TARGET_RMS_DB` (-16), `SILENCE_THRESHOLD_DB` (-40) and
`CEILING_DB` (-1).

## Tests

```bash
python -m pytest
```

The tests use generated sine waves and noise instead of real songs. They check,
among other things, that:

- the mix length equals the sum of the song lengths minus the crossfades
- no sample of the finished mix goes above 1.0
- the crossfade gains satisfy sin² + cos² = 1, and loudness really stays level
  through a crossfade (a linear fade, for comparison, dips 3 dB)
- crossfades shrink, rather than crash, when a song is too short

The MP3 export test is skipped until ffmpeg is installed.

## Project layout

```text
automix.py      command-line tool: arguments, the pipeline, the tracklist
audio_io.py     load any file as float32 stereo at 44.1 kHz; save wav/flac/mp3
processing.py   RMS and dB helpers, silence trimming, loudness, clipping protection
transitions.py  crossfade functions (equal-power for now)
mixer.py        plans the overlaps and assembles the tracks into one mix
analysis.py     placeholder for later stages: BPM, beats, key, energy
tests/          pytest tests using generated audio
```

## Known limitations

- **One very dynamic song can make the whole mix quieter.** Clipping is prevented
  by turning the *entire* mix down, which adds no distortion. But if one song has
  peaks far above its average level, normalizing it pushes those peaks up and
  everything gets turned down to fit them. The tool prints a note when this
  happens. A mastering-style limiter would fix it; that's a future upgrade.
- **RMS is only an approximation of loudness.** Bass-heavy songs can sound quieter
  than they measure. LUFS, the broadcast standard, is more accurate.
- **Everything is held in memory.** A 27-minute mix peaked at about 2 GB of RAM in
  testing, so allow roughly 4 GB for an hour.
- **No tempo awareness yet**: a crossfade between songs at different tempos sounds
  like two songs playing at once. That's what Stage 2 is for.

## Roadmap

- **Stage 2: BPM and beat detection, then beat matching.** Time-stretch songs to a
  shared tempo, start crossfades on downbeats, and measure crossfade lengths in
  bars. This plugs in by replacing `mixer.plan_crossfades`; `mixer.render_mix`
  stays the same.
- **Stage 3: key detection, then harmonic mixing.** Order songs so that neighbours
  are in compatible keys (the Camelot wheel).
- **Stage 4: energy, then set flow.** Order the set to build up and cool down,
  and choose a transition style for each pair of songs.

Stubs for the analysis functions are in `analysis.py`.

## Troubleshooting

- **`error: writing MP3 needs ffmpeg`**: install ffmpeg (see above), open a new
  terminal, and check `ffmpeg -version`. Or write a WAV instead: `-o mix.wav`.
- **A song is "skipped, couldn't decode it as audio"**: the file is damaged, or
  isn't really the format its extension says. Re-export or re-download it.
- **`pip install` fails while building numba or llvmlite**: librosa's dependencies
  can lag behind brand-new Python releases. Use the previous Python version.
- **`ModuleNotFoundError: No module named 'pyaudioop'`** (Python 3.13+): pydub
  needs the `audioop-lts` package, which `requirements.txt` installs. Re-run
  `pip install -r requirements.txt` inside the activated venv.

Tested on macOS with Python 3.14, librosa 1.0, numpy 2.5 and soundfile 0.14.
