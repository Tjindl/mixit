# mixit: auto-DJ

Turns a set of songs into one continuous DJ-style mix.

**Stage 4 (this version): structure-aware, mid-song transitions.** A real DJ
doesn't wait for a song to end. They leave it at a good moment, after a drop or as
it goes into a breakdown. They bring the next song in at a good moment, at the
start of a build so its drop hits right after the transition. And they pick a
technique to suit. automix now does the same:

- **Structure.** For each song it finds the bars (downbeats), groups them into
  8-bar phrases, and splits the song into sections: intro, build, drop (or chorus),
  breakdown, groove, outro.
- **Cue points.** It considers leaving each song at the end of a drop, the start
  of a breakdown or outro, or right before a drop (for a drop swap). It considers
  entering the next at its intro, a build, a breakdown or a drop. Every cue point
  is on a phrase boundary, and every song plays for at least `--min-play` seconds
  (default 60).
- **Three transitions:**
  - **blend**: both songs play together, beats locked, for a whole phrase (32
    beats), or half of one (16) where a song's intro or outro is too short. A
    **bass swap** keeps it clean: the incoming song's bass is cut until the
    downbeat halfway through, then the two basslines swap, so two of them never
    play at full volume together. Auto mode prefers blends wherever they work.
  - **drop swap**: the outgoing song plays its build right up to the downbeat
    where its drop would hit, and the incoming song's drop hits there instead.
  - **echo out**: the outgoing song's last beat echoes away through a closing
    low-pass filter while the next song starts. Nothing has to line up, so it's
    the fallback where tempos are too far apart to beat-match, or keys clash badly.
- **Choosing.** Every possible exit, entry and technique for every pair of songs
  is scored on local key, energy flow, clashing vocals, tempo change, and whether
  each song gets to play its drops. The best options decide the order of the set.
  Auto mode never uses the same technique three times in a row.

Everything from earlier stages is still there. Each song plays at its own tempo
and glides to a shared one only around a beat-matched transition, now wherever
that transition is (`--tempo-mode ramp`). Key and energy still shape the order
(`--order smart`, `--energy`). Songs too short to leave mid-song, or with no clear
structure, are mixed end to start as in Stage 3. `--no-beatmatch` is still
Stage 1: plain crossfades.

```bash
python automix.py songs/ -o mix.wav
```

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

| System        | Command                                                            |
| ------------- | ------------------------------------------------------------------ |
| macOS         | `brew install ffmpeg` (needs [Homebrew](https://brew.sh))          |
| Windows       | `winget install --id Gyan.FFmpeg -e`, then **open a new terminal** |
| Ubuntu/Debian | `sudo apt update && sudo apt install ffmpeg`                       |
| Fedora        | `sudo dnf install ffmpeg-free` (or `ffmpeg` from RPM Fusion)       |
| Arch          | `sudo pacman -S ffmpeg`                                            |

On Windows, `choco install ffmpeg` or `scoop install ffmpeg` work too. Check that
it's installed with `ffmpeg -version`.

### 3. Install Rubber Band (optional, but needed for tempo ramps)

Rubber Band is a time-stretching program; automix drives it through the
`pyrubberband` package that `requirements.txt` already installed.

- **Ramp mode (the default) needs Rubber Band 3.0 or newer.** It uses Rubber
  Band's R3 engine, which follows a changing tempo to within a millisecond and,
  where the tempo isn't changing, reproduces the original almost exactly, so a
  stretched transition joins seamlessly onto the untouched rest of the song.
  Without it, automix warns you and falls back to `--tempo-mode global`.
- **Global mode** works without Rubber Band, using librosa's phase vocoder, but
  Rubber Band sounds cleaner and lines up drum hits more tightly (within 2 ms in
  testing, against occasional 18 ms "flams" with the phase vocoder).

| System        | Command                                                                  |
| ------------- | ------------------------------------------------------------------------ |
| macOS         | `brew install rubberband` (or the download below, without Homebrew)      |
| Ubuntu/Debian | `sudo apt install rubberband-cli` (check the version: see below)         |
| Fedora        | `sudo dnf install rubberband`                                            |
| Arch          | `sudo pacman -S rubberband`                                              |
| Windows       | download the command-line utility (see below)                            |

Check with `rubberband --version`: it must say 3.0 or newer for ramp mode. Some
older Linux releases ship 2.x (Ubuntu 22.04, for example); there, use the download
from the Rubber Band website instead.

**Without a package manager (Windows, or macOS without Homebrew):** on the
[Rubber Band website](https://breakfastquay.com/rubberband/), download the
"Rubber Band Library command-line utility" for your system, and put the
`rubberband` program in a folder on your PATH (on macOS, `~/.local/bin` works if
it's on your PATH; on Windows, add the folder to PATH and open a new terminal).

When you run automix, the "Tempo" line says which mode and stretcher it's using.

### 4. Optional: better analysis

Everything works without these. Each one makes a part of the analysis better, and
automix uses it automatically once it's installed (songs analyzed without it are
analyzed again on the next run).

| Package | What it improves | Install |
| ------- | ---------------- | ------- |
| [essentia](https://essentia.upf.edu) | key detection (tuned on real recordings) | `pip install essentia` (macOS and Linux; no Windows builds) |
| [beat_this](https://github.com/CPJKU/beat_this) | beats and downbeats (a neural network; much more reliable than our kick-and-bass rule for finding beat 1) | `pip install https://github.com/CPJKU/beat_this/archive/main.zip` |
| [demucs](https://github.com/adefossez/demucs) | vocals: separates each song's vocals, so automix can avoid blending two singers over each other | `pip install demucs` |
| [allin1](https://github.com/mir-aidj/all-in-one) | sections and their names (a neural network trained on annotated songs) | `pip install allin1` |

The last three are neural networks. They need **PyTorch**, which pip installs
along with them (about 800 MB), and each downloads its trained model the first
time it runs (about 80 MB each). They run on the CPU:

- beat_this takes a few seconds per song.
- demucs takes 10–30 s per song. Its vocal stems are kept in a `stems` folder
  (next to `analysis_cache.json`), so each song is separated only once.
- allin1 depends on `natten` and `madmom`, which have to be compiled; at the time
  of writing madmom doesn't build on Python 3.14, so allin1 needs an older Python.

Without demucs, automix prints a note and simply doesn't score vocal clashes.

## Usage

```bash
# the best-flowing order of a folder of songs, mixed at the best cue points
python automix.py songs/ -o mix.wav

# just look: the planned order, cue points, techniques and timings, without rendering
python automix.py songs/ --plan-only

# check the cue points by eye: Audacity labels and a picture for every song
python automix.py songs/ --plan-only --cue-debug

# one technique only (falls back to echo out where it's impossible)
python automix.py songs/ -o mix.wav --transitions blend
python automix.py songs/ -o mix.wav --transitions dropswap
python automix.py songs/ -o mix.wav --transitions echo

# let each song play at least 90 s; 4-bar phrases (16 beats) instead of 8
python automix.py songs/ -o mix.wav --min-play 90 --phrase-bars 4

# compare the two tempo modes (separate folders, so each keeps its own set_plan.json)
python automix.py songs/ -o ramp/mix.wav                         # tempo changes only around transitions
python automix.py songs/ -o global/mix.wav --tempo-mode global   # every song at one tempo

# build the energy up through the set, starting with a particular song
python automix.py songs/ -o mix.wav --energy rise --start "intro.mp3"

# keep your own order: files as listed (folders alphabetically), or alphabetical
python automix.py intro.mp3 peak.flac outro.wav -o mix.wav --order given

# each song's tempo, beat count, key and energy
python automix.py songs/ --analyze-only

# check the beat detection by ear: a click on every detected beat
python automix.py songs/ -o check.wav --clicks

# Stage 1 behaviour: no tempo changes, whole songs, plain 8-second crossfades
python automix.py songs/ -o mix.wav --no-beatmatch
```

| Option                  | Meaning                                                                                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `INPUT ...`             | Folders and/or files. A folder's `.mp3`/`.wav`/`.flac` files are taken in alphabetical order (ignoring case and skipping hidden files); files in the order given. |
| `-o`, `--output`        | The mix to write: `.wav`, `.flac` (16-bit) or `.mp3` (320 kbps, needs ffmpeg). Default: `mix.wav`. `set_plan.json` is saved in the same folder.               |
| `--transitions`         | `auto` (default): the best technique for each pair, never the same one 3 times in a row. `blend`, `dropswap` or `echo`: that technique wherever it's possible, echo out elsewhere. |
| `--min-play SECONDS`    | Each song plays for at least this long. Default: 60. Songs shorter than that (plus a transition) play whole.                                                 |
| `--phrase-bars BARS`    | Bars per phrase. Default: 8 (32 beats). Cue points, and blends' starts and ends, are on phrase boundaries.                                                   |
| `--cue-debug`           | Save each song's sections and cue points as an Audacity label file and a PNG timeline, in a `cue_debug` folder next to the output.                           |
| `--order`               | `smart` (default): the order that flows best. `given`: the inputs' order. `alpha`: alphabetical by file name.                                                  |
| `--start FILENAME`      | The song that opens the set, by file name (with or without the extension). Works with every `--order`.                                                       |
| `--energy`              | Smart order's energy shape: `any` (default: just the smoothest transitions), `rise` (build up), `peak` (warm up, peak at 2/3, cool down).                      |
| `--tempo-mode`          | `ramp` (default): songs keep their own tempo, changing it only around beat-matched transitions. `global`: every song stretched to one tempo. `--bpm` implies `global`. |
| `--meet`                | Ramp mode's transition tempo: `middle` (default: both songs meet halfway, each changing half as much) or `incoming` (the outgoing song matches the next one). |
| `--ramp-beats N`        | Ramp mode: beats over which a song glides to the transition tempo, and back afterwards. Default: 16. Shortened automatically if there isn't room.            |
| `--crossfade-beats N`   | Length of beat-matched crossfades, in beats. Default: 16. Mid-song blends round it up to whole phrases (32 beats with 8-bar phrases), or use half a phrase where there's no room. |
| `--max-stretch PERCENT` | Largest tempo change allowed. Default: 8. Beat-matched transitions that would need more aren't used (echo out instead; in Stage 3 mode, a plain crossfade).  |
| `--bpm BPM`             | Global mode: the tempo for the whole mix. Default: the median of the songs' tempos.                                                                           |
| `--crossfade SECONDS`   | Length of plain crossfades: used with `--no-beatmatch`, and next to songs with no steady beat. Default: 8.                                                   |
| `--no-beatmatch`        | Stage 1 behaviour: whole songs, plain crossfades, no tempo changes (the order still uses key and energy).                                                    |
| `--plan-only`           | Print the planned set and save `set_plan.json`, without rendering any audio.                                                                                 |
| `--analyze-only`        | Print each song's tempo, beat count, key and energy, then stop.                                                                                              |
| `--clicks`              | Put a click on every detected beat in the output (songs alternate between a lower and a higher click).                                                        |

### Example

Three generated test songs, each 72 bars: intro, build, drop, breakdown, build,
drop, outro, all 8 bars long except the 16-bar drops.

```text
$ python automix.py songs/ --plan-only --cue-debug
...
Order: smart (tried all 6 orders): total cost 0.79

Planned transitions at cue points (--transitions auto; 8-bar phrases; each song plays at least 60 s):
  skyline out @ 1:31 (before drop 2) -> nightfall in @ 0:32 (drop 1) | drop swap | good
      cut on the downbeat at 127.0 BPM (skyline -0.8%, nightfall +0.8%); cost 0.37
  nightfall out @ 1:33 (before drop 2) -> lifted in @ 0:32 (drop 1) | drop swap | good
      cut on the downbeat at 125.0 BPM (nightfall -0.8%, lifted +0.8%); cost 0.43

Cue debug files written to cue_debug

Tracklist (smart order; each song at its own tempo, meeting in the middle)
    #  start  title         BPM   key  energy  plays
   1.  00:00  skyline     128.0    8A    0.52  0:00-1:31
        -> out @ 1:31 (before drop 2), nightfall in @ 0:32 (drop 1) | drop swap  [cost 0.37, good]
   2.  01:30  nightfall   126.0    9A    0.44  0:32-1:33
        -> out @ 1:33 (before drop 2), lifted in @ 0:32 (drop 1) | drop swap  [cost 0.43, good]
   3.  02:31  lifted      124.0    8A    0.51  0:32-2:21

Planned length: 04:20
```

These songs were built with clean 8-bar builds into drops of the same intensity,
which is exactly what a drop swap wants: skyline plays its second build right up to
the downbeat where its second drop would hit, and nightfall's first drop hits there
instead. (On real songs, blends usually win: see the note below.) Ask for blends
and the same songs mix like this:

```text
$ python automix.py songs/ --plan-only --transitions blend
  nightfall out @ 2:03 (end of drop 2) -> skyline in @ 0:16 (build 1) | blend | good
      32-beat blend at 127.0 BPM (nightfall +0.8%, skyline -0.8%), bass swap on beat 16; cost 0.49
  skyline out @ 2:01 (end of drop 2) -> lifted in @ 0:01 (intro) | blend | good
      32-beat blend at 126.0 BPM (skyline -1.6%, lifted +1.6%), bass swap on beat 16; cost 0.45
```

Nightfall's outro plays under skyline's build, and the bass swaps on beat 16. The
blend ends just as skyline's first drop hits.

On a real set of four Punjabi pop songs (90 to 97 BPM), auto mode planned a drop
swap between the two songs whose keys clash, then a 32-beat blend (21 s of both
songs together), then a 16-beat blend across a short outro.

Reading a transition line: **out @** is where the outgoing song is left (in its
own timeline, the time you'd see in Audacity), and the cue it's at; **in @** is
where the next song comes in, and its cue. Then come the technique and a label:
**perfect** (cost under 0.3), **good** (under 0.8) or **risky**. The indented line
gives the details: the tempo both songs meet at, how much each one changes (+ =
faster), where the bass swaps. In the tracklist, **plays** is the part of each
song that's heard.

### Checking cue points with `--cue-debug`

Structure detection isn't perfect, so `--cue-debug` lets you see what it found.
For every song in the set, a `cue_debug` folder next to the output gets:

- **`NN title.txt`**, an Audacity label file. In Audacity, open the song's
  original file, then use **File → Import → Labels...** and pick the `.txt`. A
  label track appears under the waveform, with:
  - the sections (intro, build 1, drop 1...)
  - every phrase boundary ("phrase 3")
  - the cue points automix considered ("exit? end of drop 1", "entry? build 1")
  - the ones it chose ("OUT -> next song: ...", "IN <- previous song: ...")
  - the part that plays ("plays")

  Play around the boundaries to check them.
- **`NN title.png`**, the same as a picture: the song's loudness and bass over
  time, its sections in colour, the phrase lines, the chosen cues, and a bar under
  the part that plays.

### set_plan.json

Every run (including `--plan-only`) saves the plan next to the output file. It
has:

- the settings and the order's total cost
- every song: position, start time in the mix, the part that plays, BPM, key,
  Camelot code, key confidence, energy, and its sections (with their energy and,
  with demucs, vocals)
- every transition: technique, the out and in cue points (time and name), the
  meeting tempo, each song's stretch, the length in beats and seconds, where the
  bass swaps, and the cost, broken down into its parts, with its label

Running again into the same folder replaces it.

### Checking beat detection with `--clicks`

Beat detection isn't perfect either, so `--clicks` lets you hear what it found.
Every detected beat gets a short click, at a lower pitch for songs 1, 3, 5... and a
higher pitch for songs 2, 4, 6.... The clicks follow the tempo ramps too. Listen
for:

- **clicks that drift off the kick drum within one song**: the tempo or the beat
  positions are wrong for that song
- **a flam (two clicks in quick succession) during a blend**: the two songs' beats
  aren't lined up
- **clicks a steady half-beat off the drums**: the tracker locked onto the
  off-beats

### The analysis cache

Analyzing a song takes a moment, so the results are saved in `analysis_cache.json`
(in the project folder) and reused on the next run. That covers beats, downbeats,
key, energy features, and the per-beat structure measurements. Sections are worked
out from those in an instant, so changing `--phrase-bars` doesn't need a
re-analysis.

An entry is keyed by the song's path and modification time: if you edit or replace
a song, it's analyzed again. The file is versioned, so a new version of the
analysis (like this one, which added structure) redoes everything once. To start
fresh, delete the file (and the `stems` folder, if you use demucs).

## How it works

```text
for each song:  load, resample to 44.1 kHz, stereo   audio_io.load_audio
                tempo and beats                      analysis.detect_beats (or beat_this)
                downbeats (bar starts)               analysis.find_downbeats (or beat_this)
                key and energy                       analysis.estimate_key, energy_features
                structure measurements, per beat     structure.analyze_structure (+ demucs)
                  (all reused from the cache when possible)
                trim silence, match loudness (RMS)   processing
then:           fix half/double tempo readings       analysis.octave_factors
                phrases and labelled sections        structure.sections
                cue points in every song             cues.find_cues
                score every exit/entry/technique     cues.score_option, best_options
                choose the order                     ordering.order_songs
                choose consistent cues for it        cues.choose_cues
                plan cuts, overlaps, tempo ramps     mixer.plan_cued
                stretch just the ramps and overlaps  tempo.apply_time_map
                add echo tails                       transitions.echo_tail
                join everything up                   mixer.mix_planned with transitions.bass_swap,
                                                       echo_join, equal_power_crossfade
                keep peaks below -1 dBFS             processing.clipping_gain
                write the file and set_plan.json     audio_io.save_audio
```

(Sets where no song can be cued mid-song take the Stage 3 route instead:
`mixer.plan_ramped` or `plan_beatmatched`, with songs joined end to start.)

The comments in the code explain *why* each step works the way it does. Good
places to start:

- **Structure**: the top of `structure.py` explains why dance music comes in
  8-bar phrases. "Where the music changes" covers self-similarity matrices and
  Foote's checkerboard novelty kernel. Then come the labelling rules.
- **Downbeats**: "Downbeats: where each bar starts" in `analysis.py`.
- **The transitions**: "Stage 4: DJ transitions" in `transitions.py` covers what
  low-pass and high-pass filters do, the 3-band EQ, why two basslines clash (low
  frequencies pile up and sound muddy), and the math of an echo (each repeat is
  the previous one times the feedback: a geometric decay).
- **Cue points and choosing them**: `cues.py`, including how dynamic programming
  picks a consistent set of cues for the whole set. How a drop swap finds the
  drop's real first hit is in `mixer._onset_near`.
- **Mid-song transitions in the mixer**: "Mid-song transitions" at the top of
  `mixer.py`.
- **Tempo ramps**: "Changing tempo gradually" in `tempo.py` covers time maps, why
  tempo is the derivative of beat position over time, smoothstep ramps, and
  stitching stretched regions onto untouched audio.
- **Key detection and the Camelot wheel**: "Key detection" and "The circle of
  fifths and the Camelot wheel" in `analysis.py`.
- **Ordering**: "Finding the best order" in `ordering.py` covers the Traveling
  Salesman Problem, greedy nearest neighbour and 2-opt.
- **Beat tracking, half/double tempo errors, phase vocoders, equal-power
  crossfades, decibels**: the tops of `analysis.py`, `tempo.py`, `transitions.py`
  and `processing.py`.

The processing functions are pure numpy (an array goes in, a new array comes out),
so you can experiment with them in a Python shell:

```python
from audio_io import load_audio
from analysis import detect_beats, find_downbeats
from structure import analyze_structure, sections

song = load_audio("songs/01 warmup.wav")  # float32 array, shape (samples, 2)
found = detect_beats(song)
bars = find_downbeats(song, found.beats)
features = analyze_structure(song, found.beats, bars)
for section in sections(features, phrase_bars=8).sections:
    print(f"{section.start:6.1f} - {section.end:6.1f}  {section.name}")
```

### Settings to experiment with

Settings you might want to change are constants, each with a comment saying what
it does:

- **How cue points are scored**: the block at the top of `cues.py`. `CueWeights`
  sets how much harmony, tempo, energy, vocals and best parts count. `BASE_COST`
  sets each technique's starting cost: blends 0, drop swaps 0.25, echo out 0.9, so
  blends win wherever they work and echo out only where the others are risky.
  Lower echo's cost for more echo outs, raise it for even fewer.
  `BLEND_OUT_COST`, `BLEND_IN_COST`, `ECHO_OUT_COST` and `ECHO_IN_COST` say which
  sections are good to leave and enter. Also there: `REPEAT_PENALTY` (the variety
  rule), `SHORT_BLEND_COST`, `ECHO_BEATS` (the echo's length) and `BLEND_PHRASES`.
- **How sections are found**: `NOVELTY_HALF_WIDTH`, `PEAK_THRESHOLD`,
  `DROP_ENERGY`, `BREAKDOWN_ENERGY` and `BUILD_RISE_DB` in `structure.py`.
- **The transitions**: `LOW_CUTOFF` and `HIGH_CUTOFF` (the EQ bands),
  `SWAP_FADE`, `ECHO_FEEDBACK` and `ECHO_SWEEP` in `transitions.py`;
  `ECHO_DELAY_BEATS` (1, or 0.5 for a busier half-beat echo) and
  `DROPSWAP_FADE_SECONDS` in `mixer.py`.
- **How songs are ordered**: the block at the top of `ordering.py` (the Stage 3
  costs, still used for the energy shape and in Stage 3 mode).
- **Earlier stages**: `KEY_FULL_CONFIDENCE_GAP`, `CHROMA_LOWEST` and
  `ENERGY_WEIGHTS` in `analysis.py`; `RAMP_ANCHORS_PER_BEAT` and `SETTLE` in
  `tempo.py`; `TARGET_RMS_DB`, `SILENCE_THRESHOLD_DB` and `CEILING_DB` in
  `processing.py`.

## Tests

```bash
python -m pytest
```

The tests use generated audio (sine waves, noise, chords, click tracks, and
synthetic dance tracks built from known sections) and made-up analysis data instead
of real songs. They check, among other things, that:

- **Structure.** The downbeat finder finds the bar starts of songs whose first beat
  isn't beat 1, segmentation finds every section boundary within a bar, and
  sections get the right labels. Phrases fall every 8 bars, starting where the
  sections do.
- **Cue points.** Every cue point is on a phrase boundary, and `--min-play` is
  respected, including for songs with a transition at each end. A drop swap only
  ever goes from a build into a drop, and echo out takes over when the tempos are
  too far apart. Clashing local keys and two vocals at once cost more, and auto
  mode never uses the same technique 3 times in a row.
- **Bass swap.** During a blend, the two songs' low bands are never both at full
  level: one bassline plays, then the other, swapping within 20 ms on the downbeat.
- **Drop swap.** The cut lands on the downbeat within 5 ms, with the incoming
  drop's first hit intact.
- **Echo out.** Each repeat is the last one times the feedback, the tail keeps
  fading and doesn't clip, and it ends in silence.
- **Mid-song blends.** Every beat in the overlap lines up within 10 ms, and the
  bass swap lands on a downbeat of both songs.
- **Tempo.** A tresillo song read at 4/3 of its tempo is corrected, while a steady
  beat is left alone.
- **Earlier stages.** Tempo, ramps, key detection, ordering, and the end-to-end
  mixes, as before.

Some tests run twice, once per tool, when the optional tools are installed:

- **Rubber Band**: the ramp and mid-song transition tests run with a stand-in
  stretcher (a resampler that follows the time map exactly), and again with Rubber
  Band.
- **essentia**: the key tests run with librosa's method, and again with essentia's.
- **beat_this, demucs and allin1**: `tests/test_optional_tools.py` tests each one
  only if it's installed. Every other test switches them off, so the suite stays
  fast and gives the same results whatever is installed.

The MP3 export test is skipped until ffmpeg is installed.

## Project layout

```text
automix.py      command-line tool: options, the pipeline, reports, tracklist and set_plan.json
audio_io.py     load any file as float32 stereo at 44.1 kHz; save wav/flac/mp3
analysis.py     tempo, beats and downbeats, half/double fixes, key (Camelot), energy, the cache
structure.py    phrases and sections: self-similarity, novelty, labels; vocals (demucs), allin1
cues.py         cue points, scoring every exit/entry/technique, choosing a consistent set
ordering.py     transition costs and finding the best order (brute force, greedy + 2-opt)
tempo.py        transition tempos, time maps and ramps, time-stretching (Rubber Band or phase vocoder)
mixer.py        plans the transitions (plain, beat-matched, ramped, or at cue points) and renders the mix
transitions.py  crossfades, the 3-band EQ and bass swap, the echo out
cuedebug.py     --cue-debug: Audacity label files and PNG timelines
processing.py   RMS and dB helpers, silence trimming, loudness, clipping protection, clicks
tests/          pytest tests using generated audio
```

## Known limitations

- **Section labels are educated guesses.** Without allin1, sections are labelled
  from their loudness, bass and position. That works well for dance music with
  clear builds and drops. It works less well for songs whose loudness barely
  changes (heavily compressed pop, where verse and chorus can measure alike) and
  for songs that open with their hook. Check with `--cue-debug`.
- **The downbeat fallback can pick the wrong beat.** Without beat_this, beat 1 is
  the beat with the strongest kick and bass. A four-on-the-floor track whose kick
  is equally loud on every beat, and whose bass doesn't change on the bar line,
  can fool it, and then every phrase boundary is off by a beat or more. beat_this
  fixes this.
- **Everything assumes 4/4 and a regular phrase grid.** Phrases are counted in
  8-bar blocks from one starting bar. A song with an odd-length section partway
  through drifts off the grid after it, and its later cue points are a few bars
  off.
- **Tempo can be misread.** Songs built on the tresillo (3+3+2) rhythm, common in
  Punjabi pop, reggaeton and dancehall, can read at 4/3 of their tempo (a 95 BPM
  song as 126.7). automix corrects that when the whole song clearly says so (see
  "The 3:4 trap" in `analysis.py`), but not every time. A song that's misread
  can't be beat-matched with its neighbours, so it gets echo outs. Check with
  `--clicks`, or install beat_this, which gets these right.
- **Vocal clashes are only scored with demucs**, which takes 10–30 s per song the
  first time.
- **The order and the cue points are chosen in two steps.** The order uses each
  pair's best cue points on their own. Once the order is fixed, the cue points are
  chosen together, so each song is entered before it's left and plays long enough.
  Occasionally that means a slightly different cue from the pair's best, so the
  order might not be quite optimal for the cues it ends up with.
- **Builds that come early can't be drop-swapped.** A drop swap leaves the outgoing
  song before its drop. If its only build comes within the first `--min-play`
  seconds, that would break the minimum, so it isn't offered.
- **Ramp mode needs Rubber Band 3.0+.** Without it you get global mode instead.
- **Beat-matched kicks still add up in Stage 3 mode.** Blends now swap the bass,
  but Stage 3's end-to-end crossfades don't, so their kicks can add up to 3 dB in
  the middle. The clipping protection then turns the whole mix down a little.
- **One very dynamic song can make the whole mix quieter.** Clipping is prevented
  by turning the *entire* mix down, which adds no distortion. But if one song has
  peaks far above its average level, everything gets turned down to fit them. A
  mastering-style limiter would fix it.
- **RMS is only an approximation of loudness.** LUFS, the broadcast standard, is
  more accurate.
- **Everything is held in memory.** In testing, a 10-minute mix of 5 songs peaked at about 1 GB.

## Roadmap

- ~~Stage 1: fixed-length equal-power crossfades~~
- ~~Stage 2: tempo and beat detection, then beat matching~~
- ~~Stage 3: tempo ramps around transitions; key and energy detection; smart
  ordering~~
- ~~Stage 4: downbeats, phrases and sections; mid-song cue points; blends with bass
  swaps, drop swaps and echo outs~~
- **Ideas for next:**
  - more techniques: filter sweeps, loops (repeat a bar of the outgoing song to
    stretch out a transition), and key changes (pitch-shift one song a semitone
    to fix a clash)
  - learned section labels
  - a mastering limiter
  - LUFS loudness

## Troubleshooting

- **Mostly echo outs, not blends**: the songs' tempos are too far apart to
  beat-match (more than `--max-stretch` between neighbours), or a tempo is misread.
  Check the BPMs in the output. If one is about 4/3 or 3/4 of the others, that's
  probably a misread (see Known limitations). `--transitions blend` forces blends
  wherever the tempos allow.
- **The songs are mixed end to start, with no cue points**: no song in the set can
  be left mid-song. Either they're shorter than `--min-play` plus a transition
  (try `--min-play 30`), or no clear sections were found (check with `--cue-debug`).
  Asking for a technique with `--transitions` uses cue points regardless.
- **A cue point is in the wrong place**: run with `--cue-debug` and look at the
  song in Audacity. If the bars are off, install beat_this. If the sections are
  off, try allin1, or `--phrase-bars 4` for a song built in 4-bar phrases.
- **`--cue-debug` writes no pictures**: install matplotlib (`pip install -r
  requirements.txt` does).
- **`note: demucs isn't installed`**: that's fine; vocal clashes just aren't
  scored. Install demucs (see Setup) if you'd like them to be.
- **`warning: --tempo-mode ramp needs Rubber Band 3.0 or newer`**: install Rubber
  Band (see above) and check `rubberband --version` in the same terminal. Until
  then you get global mode.
- **`error: writing MP3 needs ffmpeg`**: install ffmpeg (see above), open a new
  terminal, and check `ffmpeg -version`. Or write a WAV instead: `-o mix.wav`.
- **A song shows "no steady beat"**: beat detection found no clear pulse (ambient,
  spoken word, very rubato music). It gets plain crossfades, which is the right
  call for that kind of song.
- **A song's BPM looks wrong**: listen with `--clicks`. If it's half or double the
  real tempo and the other songs don't fix it, use `--tempo-mode global --bpm`
  with the right tempo; a reading that's half or double of `--bpm` gets corrected.
- **A key looks wrong**: install essentia (see above) for a second opinion.
- **`--start: no song called ...`**: give the file name as it is in the folder,
  with or without the extension (`--start "01 intro"` or `--start "01 intro.mp3"`).
- **A song is "skipped, couldn't decode it as audio"**: the file is damaged, or
  isn't really the format its extension says. Re-export or re-download it.
- **`pip install` fails while building numba, llvmlite or madmom**: these can lag
  behind brand-new Python releases. Use the previous Python version.
- **`ModuleNotFoundError: No module named 'pyaudioop'`** (Python 3.13+): pydub
  needs the `audioop-lts` package, which `requirements.txt` installs. Re-run
  `pip install -r requirements.txt` inside the activated venv.

Tested on macOS with Python 3.14, librosa 1.0, numpy 2.5, scipy 1.18, matplotlib
3.11, Rubber Band 4.0, essentia 2.1b6, beat_this 1.1, demucs 4.1 and PyTorch 2.15.
allin1's code path is written against allin1 1.1's API but couldn't be run here
(madmom doesn't build on Python 3.14).
